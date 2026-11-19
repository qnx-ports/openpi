#!/usr/bin/env python3
"""On the target: run QnxPolicy over a bundle's golden.msgpack (made by make_golden.py from openpi's own
Policy) and compare actions.   python3 -m openpi_qnx.check_golden <bundle_dir>"""
import sys

import numpy as np
from openpi_client import msgpack_numpy

from .policy import QnxPolicy

bundle = sys.argv[1]
g = msgpack_numpy.unpackb(open(f"{bundle}/golden.msgpack", "rb").read())
q = QnxPolicy(bundle)
worst, times = 0.0, []
for i, c in enumerate(g["cases"]):
    out = q.infer(dict(c["obs"]), noise=c["noise"])
    d = float(np.abs(out["actions"] - c["actions"]).max())
    worst = max(worst, d)
    times.append(out["policy_timing"]["infer_ms"])
    print(f"case {i}: actions {out['actions'].shape} max diff {d:.3g}  infer {out['policy_timing']['infer_ms']:.1f} ms")
ok = worst < 1e-3
print(f"{g['openpi_config']}: openpi_qnx vs openpi Policy: max diff {worst:.3g} -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
