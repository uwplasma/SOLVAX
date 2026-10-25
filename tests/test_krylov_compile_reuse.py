"""Fresh closures with the same code and shapes share one compiled solve."""

import jax
import jax.numpy as jnp
import pytest

from solvax import gcrot, gmres

jax.config.update("jax_enable_x64", True)


def _problem(seed, n=40):
    key = jax.random.PRNGKey(seed)
    a = jnp.eye(n) * 4 + 0.3 * jax.random.normal(key, (n, n)) / n**0.5
    return a, jnp.ones(n)


def _compiles(solver, n_solves, **kw):
    count = []
    with jax.log_compiles(), _capture(count):
        for i in range(n_solves):
            a, b = _problem(i)
            d = 1.0 / jnp.diag(a)
            sol = solver(lambda v, a=a: a @ v, b, precond=lambda v, d=d: d * v, rtol=1e-10, **kw)
            assert jnp.linalg.norm(a @ sol.x - b) < 1e-8
    return len(count)


class _capture:
    def __init__(self, sink):
        self.sink = sink

    def __enter__(self):
        import logging

        sink = self.sink

        class H(logging.Handler):
            def emit(self, record):
                msg = record.getMessage()
                if "Compiling" in msg and "_impl" in msg:
                    sink.append(msg)

        self.h, self.log = H(), logging.getLogger("jax")
        self.old = self.log.level
        self.log.setLevel(logging.DEBUG)
        self.log.addHandler(self.h)

    def __exit__(self, *exc):
        self.log.removeHandler(self.h)
        self.log.setLevel(self.old)


@pytest.mark.parametrize("solver", [gmres, gcrot])
def test_new_closures_compile_once(solver):
    assert _compiles(solver, 4) == 1


def test_gradient_through_captured_array():
    a, b = _problem(0)

    def loss(scale):
        return jnp.sum(gmres(lambda v: scale * (a @ v), b, rtol=1e-12, fixed_work=True).x)

    expected = -jnp.sum(jnp.linalg.solve(a, b)) / 2.0**2 * 1.0
    assert jnp.allclose(jax.grad(loss)(2.0), expected, rtol=1e-6)
