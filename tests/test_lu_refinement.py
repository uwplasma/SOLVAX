"""NumPy arithmetic checks and opt-in CUDA/AD checks; no default JAX import."""
import ast
from decimal import Decimal, localcontext
import importlib.metadata
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'src/solvax/_lu_refinement.py'


def functions(*names):
    tree = ast.parse(SOURCE.read_text())
    return [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]


def exact_fma(a, b, c):
    if hasattr(math,'fma'):
        return math.fma(a,b,c)
    with localcontext() as context:
        context.prec=200
        return float(Decimal.from_float(a)*Decimal.from_float(b)+Decimal.from_float(c))


def numpy_kernel():
    function=functions('_cuda_triangular_product')[0]
    function.body=[node for node in function.body if not isinstance(node,ast.ImportFrom)]
    position=[0,0]

    class Ref:
        def __init__(self,array): self.array=array
        @property
        def at(self): return self
        def __getitem__(self,index): return self.array,index

    def load(view,*,mask,other):
        array,index=view
        clipped=tuple(np.clip(i,0,array.shape[k]-1) for k,i in enumerate(index))
        return np.where(mask,array[clipped],other)

    def store(view,value,*,mask):
        array,index=view;rows,cols=np.broadcast_arrays(*index)
        array[rows[mask],cols[mask]]=value[mask]

    def call(kernel,*,out_shape,grid,compiler_params,name):
        def execute(*args):
            outputs=tuple(np.empty(s.shape,s.dtype) for s in out_shape)
            for position[0] in range(grid[0]):
                for position[1] in range(grid[1]):
                    kernel(*(Ref(a) for a in (*args,*outputs)))
            return outputs
        return execute

    def loop(start,end,body,carry):
        for i in range(start,end):
            updated=body(i,carry)
            assert type(updated) is type(carry), 'JAX loop carry structure changed'
            carry=updated
        return carry

    def assembly(asm,*,args,constraints,pack,result_shape_dtypes):
        assert pack==1 and constraints.startswith('=&d,=&d,')
        if asm.startswith('mul.rn'):
            a,b,low=args
            high=a*b
            fma=np.vectorize(exact_fma,otypes=[float])
            error=fma(a,b,-high)
            return [high,fma(a,low,error)]
        a,b=args
        high=a+b;z=high-a
        return [high,(a-(high-z))+(b-z)]

    env=dict(jnp=np,jax=SimpleNamespace(
        ShapeDtypeStruct=lambda shape,dtype:SimpleNamespace(shape=shape,dtype=dtype),
        lax=SimpleNamespace(fori_loop=loop,split=lambda a,sizes,axis:np.split(a,np.cumsum(sizes)[:-1],axis=axis))),
        pl=SimpleNamespace(program_id=lambda i:position[i],pallas_call=call),
        plt=SimpleNamespace(load=load,store=store,elementwise_inline_asm=assembly,CompilerParams=lambda **kw:kw))
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(SOURCE),'exec'),env)
    return env['_cuda_triangular_product']


class LuRefinementNumpyTests(unittest.TestCase):
    def test_actual_kernel_handles_both_triangles_transposes_and_edge_tiles(self):
        jax_was_loaded='jax' in sys.modules
        kernel=numpy_kernel();rng=np.random.default_rng(26)
        for n,count in ((3,2),(9,3),(17,17)):
            a=rng.normal(size=(n,n));hi=rng.normal(size=(n,count));lo=hi*2.**-55
            for lower in (False,True):
                triangle=np.tril(a,-1)+np.eye(n) if lower else np.triu(a)
                for transpose in (False,True):
                    matrix=triangle.T if transpose else triangle
                    high,low=kernel(a,hi,lo,lower=lower,transpose=transpose)
                    reference=matrix.astype(np.longdouble)@(hi.astype(np.longdouble)+lo.astype(np.longdouble))
                    actual=high.astype(np.longdouble)+low.astype(np.longdouble)
                    np.testing.assert_allclose(actual,reference,rtol=2e-17,atol=2e-17)
        self.assertEqual('jax' in sys.modules,jax_was_loaded)

    def test_product_retains_low_word_and_cancellation(self):
        kernel=numpy_kernel()
        a=np.asarray([[1e16,1.,-1e16],[0.,1.,0.],[0.,0.,1.]])
        hi,lo=kernel(a,np.ones((3,1)),np.zeros((3,1)),lower=False,transpose=False)
        self.assertEqual(float(hi[0,0]+lo[0,0]),1.)
        hi,lo=kernel(np.ones((1,1)),np.ones((1,1)),np.full((1,1),2.**-54),lower=False,transpose=False)
        self.assertEqual(float(lo[0,0]),2.**-54)

    def test_one_correction_reuses_factors_and_reduces_injected_application_error(self):
        lu=np.asarray([[2.,1.],[.25,3.]])
        perm=np.asarray([1,0]);inverse=np.argsort(perm)
        k=(np.tril(lu,-1)+np.eye(2))@np.triu(lu)
        matrix=k[inverse];rhs=np.asarray([[1.,2.],[3.,4.]])
        calls=[];epsilon=1e-4
        def native(factors,b,trans=0):
            self.assertIs(factors[0],lu);calls.append(trans)
            return np.linalg.solve(matrix.T if trans else matrix,b)*(1+epsilon)
        def bind(factor,high,low,*,lower,transpose):
            a=np.tril(factor,-1)+np.eye(2) if lower else np.triu(factor)
            if transpose: a=a.T
            wide=a.astype(np.longdouble)@(high.astype(np.longdouble)+low.astype(np.longdouble))
            hi=np.asarray(wide,dtype=float)
            return hi,np.asarray(wide-hi.astype(np.longdouble),dtype=float)
        env=dict(jnp=np,lu_solve=native,_product=SimpleNamespace(bind=bind))
        exec(compile(ast.Module(body=functions('_correct_once'),type_ignores=[]),str(SOURCE),'exec'),env)
        for trans in (0,1):
            calls.clear()
            actual=env['_correct_once'](lu,None,perm,inverse,rhs,trans)
            expected=np.linalg.solve(matrix.T if trans else matrix,rhs)
            np.testing.assert_allclose(actual,expected*(1-epsilon**2),rtol=3e-15,atol=3e-15)
            self.assertEqual(calls,[trans,trans])

    def test_inline_assembly_and_pallas_calls_match_supported_api_signatures(self):
        root=Path(importlib.metadata.distribution('jax').locate_file('jax'))
        apis={name:root/'_src'/file for name,file in (
            ('pallas_call','pallas/pallas_call.py'),
            ('elementwise_inline_asm','pallas/triton/primitives.py'),
            ('load','pallas/triton/primitives.py'),('store','pallas/triton/primitives.py'))}
        kernel=functions('_cuda_triangular_product')[0]
        for name,path in apis.items():
            function=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
            accepted={a.arg for a in function.args.posonlyargs+function.args.args+function.args.kwonlyargs}
            for call in ast.walk(kernel):
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr==name:
                    self.assertLessEqual({kw.arg for kw in call.keywords},accepted)


@unittest.skipUnless(os.environ.get('SOLVAX_TEST_LU_REFINEMENT_GPU')=='1','Explicit CUDA execution required')
class LuRefinementCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as jnp
        from solvax._lu_refinement import refined_lu_solve
        cls.jax,cls.jnp=jax,jnp;cls.solve=staticmethod(refined_lu_solve)
        jax.config.update('jax_enable_x64',True)
        if jax.default_backend()!='gpu': raise RuntimeError('CUDA required')
        print(f'[lu-refinement-tests] JAX={jax.__version__} device={jax.devices()[0]} '
              f'source={sys.modules[refined_lu_solve.__module__].__file__}',flush=True)

    def test_compiled_product_preserves_cancellation_and_low_word(self):
        from solvax._lu_refinement import _product
        jax,jnp=self.jax,self.jnp
        kernel=jax.jit(lambda a,b,low:_product.bind(a,b,low,lower=False,transpose=False))
        a=jnp.asarray([[1e16,1.,-1e16],[0.,1.,0.],[0.,0.,1.]])
        hi,lo=jax.block_until_ready(kernel(a,jnp.ones((3,1)),jnp.zeros((3,1))))
        self.assertEqual(float(hi[0,0]+lo[0,0]),1.)
        hi,lo=jax.block_until_ready(kernel(jnp.ones((1,1)),jnp.ones((1,1)),jnp.full((1,1),2.**-54)))
        self.assertEqual(float(lo[0,0]),2.**-54)

    def test_primal_transpose_vectors_and_matrix_rhs(self):
        from jax.scipy.linalg import lu_factor
        a=np.asarray([[1.,3.,.2],[4.,2.,-.1],[.2,.4,5.]])
        factors=lu_factor(self.jnp.asarray(a));rhs=np.arange(6.,dtype=float).reshape(3,2)+.5
        for value in (rhs,rhs[:,0]):
            for trans in (0,1,2):
                actual=self.jax.jit(lambda b:self.solve(factors,b,trans))(self.jnp.asarray(value))
                np.testing.assert_allclose(actual,np.linalg.solve(a.T if trans else a,value),rtol=3e-13,atol=3e-13)

    def test_forward_reverse_derivatives_and_batching(self):
        from jax.scipy.linalg import lu_factor
        jax,jnp=self.jax,self.jnp
        a=jnp.asarray([[1.,3.,.2],[4.,2.,-.1],[.2,.4,5.]])
        lu,piv=lu_factor(a);rhs=jnp.arange(6.,dtype=jnp.float64).reshape(3,2)+.5
        dlu=jnp.sin(lu)*.01;drhs=jnp.cos(rhs)*.02;bar=jnp.sin(rhs)
        function=lambda f,b:self.solve((f,piv),b)
        _,dot=jax.jit(lambda f,b,df,db:jax.jvp(function,(f,b),(df,db)))(lu,rhs,dlu,drhs)
        grads=jax.jit(jax.grad(lambda f,b:jnp.sum(function(f,b)*bar),(0,1)))(lu,rhs)
        lhs=jnp.sum(dot*bar);right=jnp.sum(grads[0]*dlu)+jnp.sum(grads[1]*drhs)
        np.testing.assert_allclose(lhs,right,rtol=2e-11,atol=2e-13)
        h=1e-5
        fd=(function(lu+h*dlu,rhs+h*drhs)-function(lu-h*dlu,rhs-h*drhs))/(2*h)
        np.testing.assert_allclose(dot,fd,rtol=3e-7,atol=3e-10)
        batched=jax.jit(jax.vmap(function,in_axes=(None,0)))(lu,jnp.stack((rhs,rhs*2)))
        np.testing.assert_allclose(batched,jnp.stack((function(lu,rhs),function(lu,rhs*2))),rtol=3e-13,atol=3e-13)


if __name__=='__main__': unittest.main()
