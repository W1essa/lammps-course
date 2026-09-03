"""
Inject MXene O-H bonds (type 2) into the equilibrated OH droplet so the
hydroxyl hydrogens can be unfrozen. 
No angles: surface groups attach to Ti via nonbonded terms only (Heinz preprint SI S9-S10).
Run from scripts: python 05_flex/patch_topology.py
"""
import os
import re
import numpy as np
from scipy.spatial import cKDTree

# paths anchored to the repo layout, independent of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))        # <root>/scripts/05_flex -> <root>
DATA = os.path.join(ROOT, "data", "mxene")
IN_FILE  = os.path.join(DATA, "equil_mxene_oh.data")
OUT_FILE = os.path.join(DATA, "equil_mxene_oh_topo.data")
TYPE_O, TYPE_H = 6, 7
OH_CUTOFF = 1.2   # A

print(f"[INFO] in : {IN_FILE}")
print(f"[INFO] out: {OUT_FILE}")

with open(IN_FILE) as f:
    lines = f.readlines()

# --- parse Atoms: id mol type q x y z ---
o_ids, o_xyz, h_ids, h_xyz = [], [], [], []
in_atoms = False
for ln in lines:
    s = ln.strip()
    if s.startswith("Atoms"):
        in_atoms = True
        continue
    if in_atoms:
        if s and s[0].isalpha():          # next section header
            break
        parts = s.split()
        if len(parts) >= 7:
            t = int(parts[2])
            if t == TYPE_O:
                o_ids.append(int(parts[0])); o_xyz.append([float(v) for v in parts[4:7]])
            elif t == TYPE_H:
                h_ids.append(int(parts[0])); h_xyz.append([float(v) for v in parts[4:7]])

print(f"[INFO] found {len(o_ids)} omx, {len(h_ids)} hoy atoms")
assert o_ids and len(o_ids) == len(h_ids), "O/H count mismatch or zero"

tree = cKDTree(np.array(h_xyz))
d, j = tree.query(np.array(o_xyz), k=1)
assert d.max() < OH_CUTOFF, f"max O-H distance {d.max():.3f} A > cutoff"
assert len(set(j.tolist())) == len(j), "an H matched to two O atoms"
pairs = [(o_ids[i], h_ids[j[i]]) for i in range(len(o_ids))]
print(f"[INFO] {len(pairs)} O-H pairs, max d = {d.max():.3f} A")

# --- patch header counters ---
out, n_old, types_patched = [], None, False
for ln in lines:
    m = re.match(r"\s*(\d+)\s+bonds\s*$", ln)
    if m:
        n_old = int(m.group(1))
        out.append(f"{n_old + len(pairs)} bonds\n")
        continue
    m = re.match(r"\s*(\d+)\s+bond types\s*$", ln)
    if m:
        out.append("2 bond types\n")
        types_patched = True
        continue
    out.append(ln)
assert n_old is not None and types_patched, "header lines not found"

# --- append new bonds at the end of the Bonds section ---
bonds_i = next(i for i, ln in enumerate(out) if ln.strip().startswith("Bonds"))
end_i = bonds_i + 1
while end_i < len(out) and not (out[end_i].strip() and out[end_i].strip()[0].isalpha()):
    end_i += 1
while not out[end_i - 1].strip():
    end_i -= 1
out[end_i:end_i] = [f"{n_old + k + 1} 2 {a} {b}\n" for k, (a, b) in enumerate(pairs)]

with open(OUT_FILE, "w") as f:
    f.writelines(out)
print(f"[OK] bonds {n_old} -> {n_old + len(pairs)}, bond types 1 -> 2")