"""NumPy edge checks; opt-in CUDA/AD checks for the fixed-order candidate.

SOLVAX_TEST_REPRODUCIBLE_GPU=1 python -m unittest discover -s tests
    -p test_reproducible_products.py -v
"""
import ast
import importlib.metadata
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def pallas_kernel_ast():
    path = ROOT/"src/solvax/_reproducible.py"
    tree = ast.parse(path.read_text())
    return path, next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "_cuda_matmul")


class ProductKernelNumpyTests(unittest.TestCase):
    def test_calls_match_installed_jax_api_without_importing_jax(self):
        """Validate actual package signatures, not a permissive mock API."""
        jax_before = sys.modules.get("jax")
        root = Path(importlib.metadata.distribution("jax").locate_file("jax"))
        _, kernel = pallas_kernel_ast()
        calls = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)]
        for name, file in (("pallas_call", "pallas/pallas_call.py"),
                           ("load", "pallas/triton/primitives.py"),
                           ("store", "pallas/triton/primitives.py"),
                           ("split", "lax/lax.py")):
            tree = ast.parse((root/"_src"/file).read_text())
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                            and node.name == name)
            positional = function.args.posonlyargs + function.args.args
            accepted = {arg.arg for arg in positional + function.args.kwonlyargs}
            for call in (node for node in calls if node.func.attr == name):
                self.assertLessEqual(len(call.args), len(positional), name)
                self.assertLessEqual({kw.arg for kw in call.keywords}, accepted, name)
                if name in ("load", "store"):
                    self.assertIsInstance(call.args[0], ast.Subscript)
                    self.assertEqual(call.args[0].value.attr, "at")
        # Ordinary array slices/gathers have no Triton lowering here. The
        # reduction uses supported equal-size split and squeeze primitives.
        for node in ast.walk(kernel):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                # Full slices below only insert broadcast axes into indices.
                indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
                for index in indices:
                    self.assertTrue((isinstance(index, ast.Constant) and index.value is None)
                                    or (isinstance(index, ast.Slice)
                                        and index.lower is index.upper is index.step is None))
        if os.environ.get("SOLVAX_TEST_REPRODUCIBLE_GPU") != "1":
            self.assertIs(sys.modules.get("jax"), jax_before)

    def test_real_kernel_indexing_and_reduction_against_extended_precision(self):
        """Execute the actual kernel body with NumPy-backed masked loads."""
        jax_before = sys.modules.get("jax")
        path, function = pallas_kernel_ast()
        function.body = [node for node in function.body if not isinstance(node, ast.ImportFrom)]
        position = [0, 0]
        params = []

        class Ref:
            def __init__(self, array):
                self.array = array

            @property
            def at(self):
                return self

            def __getitem__(self, indices):
                return self.array, indices

        def load(view, *, mask, other):
            ref, indices = view
            clipped = tuple(np.clip(index, 0, ref.shape[axis]-1)
                            for axis, index in enumerate(indices))
            return np.where(mask, ref[clipped], other)

        def store(view, values, *, mask):
            ref, indices = view
            rows, cols = np.broadcast_arrays(*indices)
            ref[rows[mask], cols[mask]] = values[mask]

        def pallas_call(kernel, *, out_shape, grid, compiler_params, name):
            params.append((compiler_params, name))

            def execute(a, b):
                result = np.empty(out_shape.shape, out_shape.dtype)
                for position[0] in range(grid[0]):
                    for position[1] in range(grid[1]):
                        kernel(Ref(a), Ref(b), Ref(result))
                return result
            return execute

        def loop(start, end, body, carry):
            for i in range(start, end):
                carry = body(i, carry)
            return carry

        def split(array, sizes, axis):
            return np.split(array, np.cumsum(sizes)[:-1], axis=axis)

        env = dict(jnp=np, jax=SimpleNamespace(lax=SimpleNamespace(fori_loop=loop, split=split),
            ShapeDtypeStruct=lambda shape, dtype: SimpleNamespace(shape=shape, dtype=dtype)),
            pl=SimpleNamespace(program_id=lambda i: position[i], pallas_call=pallas_call),
            plt=SimpleNamespace(load=load, store=store, CompilerParams=lambda **kw: kw))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), env)
        rng = np.random.default_rng(20261123)
        for m, k, n in ((1, 1, 1), (7, 3, 2), (17, 19, 33), (33, 17, 19), (16, 24, 16)):
            a, b = rng.normal(size=(m, k)), rng.normal(size=(k, n))
            result = env["_cuda_matmul"](a, b)
            reference = np.asarray(a.astype(np.longdouble) @ b.astype(np.longdouble), dtype=float)
            np.testing.assert_allclose(result, reference, rtol=3e-13, atol=3e-13)
        self.assertTrue(all(item[0] == dict(num_warps=4, num_stages=1) for item in params))
        if os.environ.get("SOLVAX_TEST_REPRODUCIBLE_GPU") != "1":
            self.assertIs(sys.modules.get("jax"), jax_before)


@unittest.skipUnless(os.environ.get("SOLVAX_TEST_REPRODUCIBLE_GPU") == "1",
                     "CUDA execution is explicitly enabled on the large machine")
class ProductKernelCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as jnp
        from solvax._reproducible import schur_product
        cls.jax, cls.jnp = jax, jnp
        cls.product = staticmethod(schur_product)
        jax.config.update("jax_enable_x64", True)
        if not jax.devices("gpu"):
            raise unittest.SkipTest("CUDA device required")

    def arrays(self, m=33, k=17, n=19):
        rng = np.random.default_rng(23)
        return tuple(self.jnp.asarray(x) for x in
                     (rng.normal(size=(m, k)), rng.normal(size=(k, n))))

    def test_nested_jit_and_independent_callable_match(self):
        a, b = self.arrays()
        direct = self.jax.jit(self.product)
        nested = self.jax.jit(lambda a, b: self.jax.jit(self.product)(a, b))
        x, y = direct(a, b), nested(a, b)
        np.testing.assert_array_equal(x, y)
        np.testing.assert_allclose(x, np.asarray(a) @ np.asarray(b), rtol=3e-13, atol=3e-13)
        np.testing.assert_array_equal(direct(a, b), x)
        executable = direct.lower(a, b).compile()
        self.assertNotIn('__cublas$lt$matmul', executable.as_text())

    def test_jvp_vjp_and_vmap_use_the_ordinary_product_derivative(self):
        jax, jnp = self.jax, self.jnp
        a, b = self.arrays()
        da, db = jnp.sin(a), jnp.cos(b)
        _, tangent = jax.jit(lambda a, b, da, db: jax.jvp(self.product, (a, b), (da, db)))(a, b, da, db)
        np.testing.assert_allclose(tangent, np.asarray(da) @ np.asarray(b) + np.asarray(a) @ np.asarray(db),
                                   rtol=4e-13, atol=4e-13)
        bars = jnp.sin(jnp.arange(a.shape[0]*b.shape[1], dtype=a.dtype)).reshape(a.shape[0], b.shape[1])
        left, right = jax.jit(jax.grad(lambda a, b: jnp.sum(self.product(a, b)*bars), (0, 1)))(a, b)
        np.testing.assert_allclose(left, np.asarray(bars) @ np.asarray(b.T), rtol=4e-13, atol=4e-13)
        np.testing.assert_allclose(right, np.asarray(a.T) @ np.asarray(bars), rtol=4e-13, atol=4e-13)
        for axes, aa, bb in (([0, None], jnp.stack((a, da)), b),
                             ([None, 0], a, jnp.stack((b, db))),
                             ([0, 0], jnp.stack((a, da)), jnp.stack((b, db)))):
            actual = jax.jit(jax.vmap(self.product, in_axes=axes))(aa, bb)
            expected = np.stack([np.asarray(aa[i] if axes[0] == 0 else aa) @
                                 np.asarray(bb[i] if axes[1] == 0 else bb) for i in range(2)])
            np.testing.assert_allclose(actual, expected, rtol=4e-13, atol=4e-13)

    def test_vector_empty_and_non_fp64_products(self):
        a, b = self.arrays()
        for name, aa, bb in (("vector", a, b[:, 0]), ("empty", a[:, :0], b[:0]),
                             ("float32", a.astype('float32'), b.astype('float32'))):
            with self.subTest(case=name):
                if name == "float32":
                    # NumPy computes at full FP32 precision. Match that policy
                    # only for this fallback/reference check; library defaults
                    # and the fixed FP64 kernel are unchanged.
                    with self.jax.default_matmul_precision("highest"):
                        actual = self.jax.jit(self.product)(aa, bb)
                    tolerance = 3e-5
                else:
                    actual = self.jax.jit(self.product)(aa, bb)
                    tolerance = 3e-13
                np.testing.assert_allclose(actual, np.asarray(aa) @ np.asarray(bb),
                                           rtol=tolerance, atol=tolerance)

    def test_generated_solver_and_its_parameter_gradient(self):
        from solvax.direct import block_thomas_truncated_fn
        jax, jnp = self.jax, self.jnp
        a, _ = self.arrays(7, 17, 1)
        lower = jnp.stack((jnp.zeros((17, 17)),)*7)
        lower = lower.at[1:].set(jnp.eye(17)*.1)
        upper = lower[::-1]
        diagonal = jnp.tile(jnp.eye(17)[None], (7, 1, 1))*3.
        rhs = a[:3].reshape(3, 17, 1)

        def solve(d):
            return block_thomas_truncated_fn(lambda k: (lower[k], d[k], upper[k]), 7, rhs, 3)

        value = jax.jit(solve)(diagonal)
        dense = np.zeros((7*17, 7*17))
        for k in range(7):
            dense[k*17:(k+1)*17, k*17:(k+1)*17] = np.asarray(diagonal[k])
            if k:
                dense[k*17:(k+1)*17, (k-1)*17:k*17] = np.asarray(lower[k])
                dense[(k-1)*17:k*17, k*17:(k+1)*17] = np.asarray(upper[k-1])
        full_rhs = np.zeros((7, 17, 1))
        full_rhs[:3] = np.asarray(rhs)
        expected = np.linalg.solve(dense, full_rhs.reshape(-1, 1)).reshape(7, 17, 1)[:3]
        np.testing.assert_allclose(value, expected, rtol=4e-13, atol=4e-13)
        loss = lambda d: jnp.sum(solve(d)**2)
        gradient = jax.jit(jax.grad(loss))(diagonal)
        direction = jnp.zeros_like(diagonal).at[2, 3, 4].set(1.)
        h = 1e-5
        fd = (loss(diagonal+h*direction)-loss(diagonal-h*direction))/(2*h)
        np.testing.assert_allclose(gradient[2, 3, 4], fd, rtol=2e-7, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
