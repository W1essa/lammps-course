#!/usr/bin/env python3
import os
import numpy as np

# =====================================================================
# --- CONFIGURATION PARAMETERS ---
# =====================================================================
# python 01_substrate/generate_Mxene.py 

# Termination of the Ti3C2Tx surface:
#   'OH'  — fully hydroxyl-terminated (Ti3C2(OH)2)
#   'F'   — fully fluorine-terminated (Ti3C2F2)
#   'MIX' — random OH/F mixture with OH_FRACTION, fixed RANDOM_SEED
TERMINATION  = 'F'
OH_FRACTION  = 0.5
RANDOM_SEED  = 42

TARGET_L  = 400.0   # A — target lateral (X, Y) size of the sheet
A_LATTICE = 3.088   # A — hex lattice constant, Ti3C2(OH)2 DFT (Khazaei 2013 SI)

# Layer heights relative to bottom H, A (from Khazaei SI CIF, z = frac_z * c*sin(beta))
Z = {
    'H_bot':  0.00,
    'O_bot':  0.98,
    'Ti_bot': 2.25,
    'C_bot':  3.34,
    'Ti_mid': 4.63,
    'C_top':  5.91,
    'Ti_top': 7.00,
    'O_top':  8.27,
    'H_top':  9.25,
}
Z_F_OFFSET = 1.25   # A — |dz| of F vs outer Ti (Ti3C2F2 CIF, Khazaei SI: 8.143-6.895)

# In-plane sublattices of the hex cell (fractional): A=(0,0), B=(1/3,2/3), C=(2/3,1/3).
# Ti planes stack fcc-like A-B-C; carbons fill octahedral holes.
SUBLATTICE = {
    'Ti_bot': (0.0, 0.0),          # A
    'C_bot':  (2.0/3.0, 1.0/3.0),  # C
    'Ti_mid': (1.0/3.0, 2.0/3.0),  # B
    'C_top':  (0.0, 0.0),          # A
    'Ti_top': (2.0/3.0, 1.0/3.0),  # C
}
# Termination hollow site, model I (no C underneath). Verify once in OVITO
# against the Khazaei CIF block; flip to the other hollow if it disagrees.
TERM_SITE_BOT = (1.0/3.0, 2.0/3.0)   # above Ti_mid column
TERM_SITE_TOP = (1.0/3.0, 2.0/3.0)

# IFF atom types, charges (e), masses (amu) — Winetrout et al. 2025, Table S1
TYPES = {
    'ti1': (1, +0.35,   47.867),
    'ti2': (2, +0.675,  47.867),
    'cx':  (3, -0.35,   12.011),
    'omx': (4, -0.7875, 15.999),
    'hoy': (5, +0.2875,  1.008),
    'fx':  (6, -0.5,    18.998),
}

# =====================================================================
# --- GENERATION FUNCTIONS ---
# =====================================================================

def hex_vectors(a):
    """Hexagonal in-plane lattice vectors."""
    a1 = np.array([a, 0.0])
    a2 = np.array([a / 2.0, a * np.sqrt(3.0) / 2.0])
    return a1, a2


def lattice_sites(target_l, a):
    """(i, j) integer pairs whose cell origin falls into [0, L]^2 footprint."""
    a1, a2 = hex_vectors(a)
    n = int(target_l / a) + 6
    sites = []
    for i in range(-n, n):
        for j in range(-n, n):
            r = i * a1 + j * a2
            if -1e-6 <= r[0] <= target_l and -1e-6 <= r[1] <= target_l:
                sites.append(r)
    return np.array(sites), (a1, a2)


def frac_to_cart(frac, a1, a2):
    return frac[0] * a1 + frac[1] * a2


def build_mxene_sheet(termination=TERMINATION):
    """
    Build a flat finite Ti3C2Tx flake ~TARGET_L x TARGET_L.

    Returns:
        atoms — list of (type_key, x, y, z)
    Per-cell columns are charge-neutral (OH group = -0.5 e = F charge),
    so a finite flake stays neutral for any OH/F mixture.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    origins, (a1, a2) = lattice_sites(TARGET_L, A_LATTICE)
    atoms = []

    core = [('ti2', 'Ti_bot'), ('cx', 'C_bot'), ('ti1', 'Ti_mid'),
            ('cx', 'C_top'), ('ti2', 'Ti_top')]

    for r0 in origins:
        column = []
        for tkey, layer in core:
            xy = r0 + frac_to_cart(SUBLATTICE[layer], a1, a2)
            column.append((tkey, xy[0], xy[1], Z[layer]))

        # bottom / top termination, independent random draw per surface site
        for side, site, z_ti, sign in (
            ('bot', TERM_SITE_BOT, Z['Ti_bot'], -1.0),
            ('top', TERM_SITE_TOP, Z['Ti_top'], +1.0),
        ):
            xy = r0 + frac_to_cart(site, a1, a2)
            if termination == 'OH' or (termination == 'MIX' and rng.random() < OH_FRACTION):
                z_o = Z['O_bot'] if side == 'bot' else Z['O_top']
                z_h = Z['H_bot'] if side == 'bot' else Z['H_top']
                column.append(('omx', xy[0], xy[1], z_o))
                column.append(('hoy', xy[0], xy[1], z_h))
            else:
                column.append(('fx', xy[0], xy[1], z_ti + sign * Z_F_OFFSET))

        # rectangle crop by WHOLE columns only (keeps charge neutrality)
        if all(-1e-6 <= x <= TARGET_L and -1e-6 <= y <= TARGET_L for _, x, y, _ in column):
            atoms.extend(column)

    return atoms


# =====================================================================
# --- SANITY CHECKS ---
# =====================================================================

def check_structure(atoms):
    """Print total charge, thickness and per-type counts; hard-fail on charge."""
    q_tot = sum(TYPES[t][1] for t, *_ in atoms)
    zs = np.array([z for *_, z in atoms])
    counts = {}
    for t, *_ in atoms:
        counts[t] = counts.get(t, 0) + 1

    print(f"[CHECK] atoms          : {len(atoms)}")
    print(f"[CHECK] total charge   : {q_tot:+.6f} e")
    print(f"[CHECK] slab thickness : {zs.max() - zs.min():.3f} A (expect ~9.3 OH / ~7.2 F)")
    print(f"[CHECK] per type       : {counts}")

    if abs(q_tot) > 1e-6:
        raise ValueError("Non-zero total charge — termination assignment is broken.")


# =====================================================================
# --- FILE WRITING ---
# =====================================================================

def write_lammps_full(atoms, filename, padding=20.0):
    """
    LAMMPS data file, atom_style full, WITH per-atom charges.
    Types 1-6 are always declared (stable ids across terminations);
    water types are appended later at the assembly step.
    """
    pos = np.array([[x, y, z] for _, x, y, z in atoms])
    mins = pos.min(axis=0) - padding
    maxs = pos.max(axis=0) + padding

    with open(filename, "w") as f:
        f.write("LAMMPS data: Ti3C2Tx MXene (full, IFF charges)\n\n")
        f.write(f"{len(atoms)} atoms\n\n")
        f.write("6 atom types\n\n")
        f.write(f"{mins[0]:.6f} {maxs[0]:.6f} xlo xhi\n")
        f.write(f"{mins[1]:.6f} {maxs[1]:.6f} ylo yhi\n")
        f.write(f"{mins[2]:.6f} {maxs[2]:.6f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for key in ('ti1', 'ti2', 'cx', 'omx', 'hoy', 'fx'):
            tid, _, m = TYPES[key]
            f.write(f"{tid} {m}\n")
        f.write("\nAtoms # full\n\n")
        for i, (t, x, y, z) in enumerate(atoms, start=1):
            tid, q, _ = TYPES[t]
            f.write(f"{i} 1 {tid} {q:.4f} {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"[INFO] LAMMPS data written to {filename}")


# =====================================================================
# --- MAIN ---
# =====================================================================

if __name__ == "__main__":
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    # repo layout: <root>/scripts/01_substrate/generate_Mxene.py
    # data goes to: <root>/data/mxene/
    root_dir = os.path.dirname(os.path.dirname(base_dir))
    output_dir = os.path.join(root_dir, "data", "mxene")
    os.makedirs(output_dir, exist_ok=True)
    tag = TERMINATION.lower() if TERMINATION != 'MIX' else f"mix{int(OH_FRACTION*100)}"
    output_file = os.path.join(output_dir, f"mxene_{tag}_initial.data")

    print(f"[INFO] Generating Ti3C2Tx sheet, termination = {TERMINATION}...")

    mxene_atoms = build_mxene_sheet()
    check_structure(mxene_atoms)
    write_lammps_full(mxene_atoms, output_file)

    pos = np.array([[x, y, z] for _, x, y, z in mxene_atoms])
    dims = pos.max(axis=0) - pos.min(axis=0)
    print(f"\n--- MXene Structure Dimensions ---")
    print(f"Size  : {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} A")
    print(f"Atoms : {len(mxene_atoms)}")
    print("----------------------------------")