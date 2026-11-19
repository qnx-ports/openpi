#!/usr/bin/env python3
"""reference.npz -> ref/<name>.bin (raw, C order) + ref/manifest.txt ("name dtype d0 d1 ...") for the C++ harness."""
import os, sys
import numpy as np

d = sys.argv[1]
ref = np.load(f"{d}/reference.npz")
os.makedirs(f"{d}/ref", exist_ok=True)
with open(f"{d}/ref/manifest.txt", "w") as mf:
    for k in ref.files:
        a = np.ascontiguousarray(ref[k])
        dt = {"float32": "f32", "bool": "bool", "int64": "i64"}[str(a.dtype)]
        a.tofile(f"{d}/ref/{k}.bin")
        mf.write(" ".join([k, dt, *map(str, a.shape)]) + "\n")
print(open(f"{d}/ref/manifest.txt").read())
