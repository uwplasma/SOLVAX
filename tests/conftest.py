"""Force a multi-device CPU platform before JAX initializes.

The sharding tests need several devices; emulating eight CPU devices here (the
module is imported before any test imports jax) makes them runnable on every CI
machine. Single-device semantics of all other tests are unchanged: arrays stay
committed to device 0 unless a test asks for a sharding.
"""

import os

_FLAG = "--xla_force_host_platform_device_count=8"
if _FLAG not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _FLAG).strip()
