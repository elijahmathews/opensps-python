"""Per-model cost of the batch from Python, in the configuration of the core's
tests/benchmark_csp_batch ("basic": MIST + MILES + DL07, tau model, dust2 = 0.3, dust
emission; tau, tage, U_min, dust2 and dust1 vary over the batch), and of the same call made
directly through ctypes (no Python wrapper), to isolate the wrapper's overhead.

Usage: python benchmarks/bench_batch.py [n ...]   (default 1 100 1000)
"""

import ctypes
import sys
import time

import numpy as np

import opensps

sizes = [int(a) for a in sys.argv[1:]] or [1, 100, 1000]
ctx = opensps.Context("mist", "miles", "DL07")
ctx.set_params(sfh=1, tau=1.0, tage=10.0, dust2=0.3, compute_indices=0)


def params(n):
    i = np.arange(1, n + 1)
    return dict(tau=0.3 + 0.05 * (i % 97), tage=1.0 + 0.1 * (i % 113),
                duste_umin=1.0 + 0.2 * (i % 71), dust2=0.05 + 0.01 * (i % 89),
                dust1=0.02 * (i % 53))


# Direct ctypes call of the same entry point in the same library
lib = ctypes.CDLL(opensps._core.__file__)
D = ctypes.POINTER(ctypes.c_double)

for n in sizes:
    p = params(n)
    reps = max(3, min(200, 20000 // n))
    out = ctx.compute_csp_batch(p, spec=False)  # warm-up (builds the SSP basis)
    t0 = time.perf_counter()
    for _ in range(reps):
        ctx.compute_csp_batch(p, spec=False, out=out)
    mags = (time.perf_counter() - t0) / reps / n * 1e6
    out = ctx.compute_csp_batch(p, spec=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ctx.compute_csp_batch(p, spec=True, out=out)
    spec = (time.perf_counter() - t0) / reps / n * 1e6

    names = (ctypes.c_char_p * len(p))(*[k.encode() for k in p])
    arrs = [np.ascontiguousarray(v, float) for v in p.values()]
    vals = (D * len(p))(*[a.ctypes.data_as(D) for a in arrs])
    m = np.empty((n, ctx.nbands))
    st = ctypes.c_int(0)
    args = (ctypes.c_int(ctx.handle), ctypes.c_int(n), ctypes.c_int(len(p)), names, vals, None,
            m.ctypes.data_as(D), None, None, None, None, None, None, None, ctypes.byref(st))
    lib.fsps_context_compute_csp_batch(*args)
    t0 = time.perf_counter()
    for _ in range(reps):
        lib.fsps_context_compute_csp_batch(*args)
    direct = (time.perf_counter() - t0) / reps / n * 1e6
    assert st.value == 0
    print(f"n={n:6d}  python mags {mags:9.1f} us/model   python mags+spec {spec:9.1f}   "
          f"direct C (ctypes) mags {direct:9.1f}")
