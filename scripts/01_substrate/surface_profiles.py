"""
surface_profiles.py
===================
Periodic surface topography for Ti3C2Tx MXene substrates.

WHY GROOVES ALWAYS RUN ALONG Y
------------------------------
The contact-angle pipeline averages the water density over y, so the substrate
cross-section must be identical at every y. Height therefore depends on x only,
h = h(x). Grooves running along x would smear both the Gibbs contour and the
substrate baseline and break the existing analysis.

MODES
-----
  ripple   One sheet, corrugated. Whole atomic ROWS (atoms sharing an x value)
           move together, so nothing inside a row is torn apart and no atom is
           deleted -> total charge is unchanged by construction.
           Represents a buckled/wrinkled monolayer.

  terrace  Several flat sheets stacked with x-dependent coverage (staircase).
           Closer to a real restacked MXene coating. Material IS removed, so
           cuts are snapped to charge-neutral row boundaries.

Both modes leave every sheet internally undistorted in-plane: no bond lengths
inside a hydroxyl are changed (O and H share the same x, so they always travel
in the same row).

Usage: see the HOOK section at the bottom of this file.
"""

import math
import numpy as np
from scipy.spatial import cKDTree

# IFF charges (Heinz group). A full Ti3C2Tx formula unit
# (1 ti1 + 2 ti2 + 2 cx + 2 T) sums to exactly zero for both OH and F.
IFF_CHARGE = {
    "ti1": 0.35,
    "ti2": 0.675,
    "cx": -0.35,
    "omx": -0.7875,
    "hoy": 0.2875,
    "fx": -0.5,
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# period / amplitude in Angstrom. Keep amplitude/period around 0.3 to match the
# aspect ratio of real LIPSS, and keep period small enough that the droplet
# spans at least 2-3 periods (droplet base is ~55-70 A, so period <= ~25 A).
PROFILE = dict(
    mode="ripple",        # "flat" | "ripple" | "terrace"
    period=25.0,          # groove period along x
    amplitude=8.0,        # peak-to-valley height
    interlayer=12.4,      # sheet-to-sheet spacing, terrace mode only
    row_tol=0.05,         # x tolerance when grouping atoms into rows
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def height(x, period, amplitude, x0=0.0):
    """Sinusoidal profile, mean 0, peak-to-valley = amplitude."""
    return 0.5 * amplitude * math.cos(2.0 * math.pi * (x - x0) / period)


def total_charge(atoms):
    return sum(IFF_CHARGE[a[0]] for a in atoms)


def wenzel_r(period, amplitude, n=4000):
    """Roughness factor r = true area / projected area for a 1-D corrugation."""
    x = np.linspace(0.0, period, n, endpoint=False)
    dh = -math.pi * amplitude / period * np.sin(2.0 * np.pi * x / period)
    return float(np.mean(np.sqrt(1.0 + dh ** 2)))


def wenzel_r_terrace(period, amplitude):
    """Staircase: flat terraces plus vertical walls, up and down once per period."""
    return 1.0 + 2.0 * amplitude / period


def _rows(atoms, tol):
    """
    Group atom indices into rows along y. A row is the set of atoms sharing one
    x value: the smallest piece of the sheet that can be displaced vertically
    without distorting anything inside it.
    """
    xs = np.array([a[1] for a in atoms], dtype=float)
    order = np.argsort(xs)
    rows, cur, x_ref = [], [int(order[0])], xs[order[0]]
    for k in order[1:]:
        if xs[k] - x_ref <= tol:
            cur.append(int(k))
        else:
            rows.append(cur)
            cur = [int(k)]
            x_ref = xs[k]
    rows.append(cur)
    row_x = np.array([xs[r].mean() for r in rows])
    return rows, row_x


def _row_charges(atoms, rows):
    return np.array([sum(IFF_CHARGE[atoms[i][0]] for i in r) for r in rows])


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_oh_bonds(atoms, cutoff=1.2):
    """Every hydroxyl O must still find its H within the bond cutoff."""
    o = np.array([[a[1], a[2], a[3]] for a in atoms if a[0] == "omx"])
    h = np.array([[a[1], a[2], a[3]] for a in atoms if a[0] == "hoy"])
    if len(o) == 0 or len(h) == 0:
        print("[CHECK] no hydroxyls present, O-H check skipped")
        return
    d, _ = cKDTree(h).query(o, k=1)
    print(f"[CHECK] max O-H distance after profiling: {d.max():.3f} A")
    if d.max() >= cutoff:
        raise ValueError("profiling broke an O-H bond - check row grouping tolerance")


def report(atoms, label=""):
    pos = np.array([[a[1], a[2], a[3]] for a in atoms])
    q = total_charge(atoms)
    print(f"[INFO] {label}: {len(atoms)} atoms, "
          f"x {pos[:, 0].min():.1f}..{pos[:, 0].max():.1f}, "
          f"z {pos[:, 2].min():.1f}..{pos[:, 2].max():.1f} A, "
          f"net charge {q:+.6f}")
    if abs(q) > 1e-6:
        print("[WARN] net charge is not zero - PPPM will complain")


# ---------------------------------------------------------------------------
# Mode: ripple (single corrugated sheet)
# ---------------------------------------------------------------------------
def apply_ripple(atoms, period, amplitude, row_tol=0.05):
    """
    Bend the sheet onto a sinusoid by ARC LENGTH: the flat x coordinate is the
    arc length along the corrugated surface, so atoms per unit of SURFACE stay
    exactly what they were on the flat sheet.

    A naive z -> z + h(x) keeps the footprint and stretches the sheet over a
    longer surface, diluting the surface groups by the roughness factor r
    (~26% at amplitude/period = 0.32). That dilution would be indistinguishable
    from a topography effect in the contact angle.

    Side effect: the projected footprint shrinks by ~r. Raise TARGET_L in
    generate_Mxene.py if you want the same footprint as the flat baseline.
    """
    rows, row_x = _rows(atoms, row_tol)

    # s(x) on a fine grid, then invert to x(s)
    xg = np.linspace(0.0, row_x.max() + period, 200001)
    hp = -math.pi * amplitude / period * np.sin(2.0 * math.pi * xg / period)
    ds = np.sqrt(1.0 + hp ** 2)
    sg = np.concatenate(([0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(xg))))

    s_row = row_x - row_x.min()          # flat coordinate == arc length
    x_new = np.interp(s_row, sg, xg)
    dz = 0.5 * amplitude * np.cos(2.0 * np.pi * x_new / period)
    dx = x_new - s_row                   # lateral shift of the whole row

    out = list(atoms)
    for r, sx, sz in zip(rows, dx, dz):
        for i in r:
            t, x, y, z = out[i]
            out[i] = (t, x + sx, y, z + sz)

    arc, proj = s_row.max(), x_new.max() - x_new.min()
    print(f"[INFO] ripple: {len(rows)} rows, arc {arc:.1f} A -> projected "
          f"{proj:.1f} A (r_actual = {arc / proj:.4f})")
    print(f"[INFO] max vertical offset between neighbouring rows: "
          f"{np.abs(np.diff(dz)).max():.2f} A")
    print(f"[INFO] Wenzel r = {wenzel_r(period, amplitude):.4f}")
    check_oh_bonds(out)
    report(out, "rippled sheet")
    return out


# ---------------------------------------------------------------------------
# Mode: terrace (stacked sheets, staircase coverage)
# ---------------------------------------------------------------------------
def _snap_neutral(row_q, i0, i1, max_shift=16, tol=1e-6):
    """Nudge the right-hand cut so the kept block of rows carries zero charge."""
    if abs(row_q[i0:i1].sum()) < tol:
        return i0, i1
    for d in range(1, max_shift + 1):
        for j in (i1 + d, i1 - d):
            if i0 < j <= len(row_q) and abs(row_q[i0:j].sum()) < tol:
                return i0, j
    return None


def stack_terraces(atoms, period, amplitude, interlayer, row_tol=0.05):
    """
    Layer 0 keeps full coverage (no hole through to the vacuum). Layer k >= 1 is
    present only where the target height reaches k * interlayer, producing a
    ridge made of whole flat sheets. Cuts fall between rows, never through one,
    and each kept block is charge-neutral.
    """
    rows, row_x = _rows(atoms, row_tol)
    row_q = _row_charges(atoms, rows)
    pos_z = np.array([a[3] for a in atoms])
    thickness = pos_z.max() - pos_z.min()
    gap = interlayer - thickness
    print(f"[INFO] terrace: sheet thickness {thickness:.2f} A, interlayer "
          f"{interlayer:.2f} A -> inter-sheet gap {gap:.2f} A")
    if gap < 2.0:
        print("[WARN] gap below ~2 A: neighbouring sheets will overlap sterically")

    n_layers = int(math.floor(amplitude / interlayer)) + 1
    print(f"[INFO] terrace: {n_layers} layers "
          f"(amplitude {amplitude:.1f} A / interlayer {interlayer:.1f} A)")

    out = list(atoms)  # layer 0, full coverage

    # target height above the valley floor, 0..amplitude
    h_target = np.array([0.5 * amplitude * (1.0 + math.cos(2.0 * math.pi * x / period))
                         for x in row_x])

    for k in range(1, n_layers):
        keep = h_target >= k * interlayer
        if not keep.any():
            continue
        # contiguous intervals of kept rows
        idx = np.flatnonzero(keep)
        splits = np.flatnonzero(np.diff(idx) > 1)
        blocks = np.split(idx, splits + 1)
        n_added = 0
        for b in blocks:
            snapped = _snap_neutral(row_q, int(b[0]), int(b[-1]) + 1)
            if snapped is None:
                print(f"[WARN] layer {k}: could not make a block neutral, skipped")
                continue
            i0, i1 = snapped
            for j in range(i0, i1):
                for i in rows[j]:
                    t, x, y, z = atoms[i]
                    out.append((t, x, y, z + k * interlayer))
                    n_added += 1
        print(f"[INFO] layer {k}: +{n_added} atoms in {len(blocks)} block(s)")

    print(f"[INFO] Wenzel r = {wenzel_r_terrace(period, amplitude):.4f}")
    check_oh_bonds(out)
    report(out, "terraced stack")
    return out


# ---------------------------------------------------------------------------
# Dispatcher + filename tag
# ---------------------------------------------------------------------------
def apply_profile(atoms, profile=None):
    p = dict(PROFILE) if profile is None else dict(profile)
    mode = p.get("mode", "flat")
    if mode == "flat":
        report(atoms, "flat sheet")
        return atoms
    if mode == "ripple":
        return apply_ripple(atoms, p["period"], p["amplitude"], p.get("row_tol", 0.05))
    if mode == "terrace":
        return stack_terraces(atoms, p["period"], p["amplitude"],
                              p["interlayer"], p.get("row_tol", 0.05))
    raise ValueError(f"unknown profile mode: {mode}")


def tag_suffix(profile=None):
    """Filename fragment so profiled sheets never overwrite the flat baseline."""
    p = dict(PROFILE) if profile is None else dict(profile)
    mode = p.get("mode", "flat")
    if mode == "flat":
        return ""
    short = {"ripple": "rip", "terrace": "ter"}[mode]
    return f"_{short}{int(round(p['period']))}a{int(round(p['amplitude']))}"