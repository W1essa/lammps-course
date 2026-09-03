"""
Contact Angle Analysis of a Water Droplet
=========================================
Analyzes 2-D density fields produced by LAMMPS chunk/ave.

PRIMARY METHOD: Local tangent fitting with iterative sigma-clipping.
SECONDARY METHOD: Geometric angle check (theta = 2 * arctan(h/a)).

MXene notes:
- substrate top is found across all substrate types (OH: top atom is H, type 7;
  F: top atom is F, type 8), so configs carry a tuple of types
- bulk density is estimated excluding the first bulk_exclude_z A above the
  substrate: hydrophilic MXene has strong adsorption layering (rho up to ~2)
  that would bias the bulk estimate
- analysis runs per production block to check the contact angle plateau
"""

import math
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =============================================================================
# 1. CONFIGURATION & DATA STRUCTURES
# =============================================================================

@dataclass
class SystemConfig:
    """Stores paths and specific behavior flags for a simulation system."""
    name: str
    density_file: str            # may contain {block} placeholder
    data_file: str
    out_dir: str
    is_flat: bool
    substrate_types: Tuple[int, ...] = (3,)
    z_win: Optional[Tuple[float, float]] = None   # per-system window override
    n_win_shifts: Optional[int] = None            # 1 = no auto-shifting

CONFIGS: Dict[str, SystemConfig] = {
    "graphene": SystemConfig(
        name="Graphene",
        density_file="../data/density_graphene.dat",
        data_file="../data/equilibrated_graphene.data",
        out_dir="../output",
        is_flat=True,
        substrate_types=(3,)
    ),
    "graphite": SystemConfig(
        name="Graphite",
        density_file="../data/density_graphite.dat",
        data_file="../data/equilibrated_graphite.data",
        out_dir="../output",
        is_flat=True,
        substrate_types=(3,)
    ),
    "structured_graphite": SystemConfig(
        name="Structured Graphite",
        density_file="../data/density_structured_graphite.dat",
        data_file="../data/equilibrated_structured_graphite.data",
        out_dir="../output",
        is_flat=False,
        substrate_types=(3,)
    ),
    "mxene_oh": SystemConfig(
        name="MXene Ti$_3$C$_2$(OH)$_2$",
        density_file="../data/mxene/density_oh_{block}.dat",
        data_file="../data/mxene/equil_mxene_oh.data",
        out_dir="../output/mxene",
        is_flat=True,
        substrate_types=(3, 4, 5, 6, 7, 8)
    ),
    "mxene_f": SystemConfig(
        name="MXene Ti$_3$C$_2$F$_2$",
        density_file="../data/mxene/density_f_{block}.dat",
        data_file="../data/mxene/equil_mxene_f.data",
        out_dir="../output/mxene",
        is_flat=True,
        z_win=(1.5, 8.0),
        n_win_shifts=1,
        substrate_types=(3, 4, 5, 6, 7, 8)
    ),
    "mxene_ohflex": SystemConfig(
        name="MXene Ti$_3$C$_2$(OH)$_2$ (mobile H)",
        density_file="../data/mxene/density_ohflex_{block}.dat",
        data_file="../data/mxene/equil_mxene_ohflex.data",
        out_dir="../output/mxene",
        is_flat=True,
        substrate_types=(3, 4, 5, 6, 7, 8)
    ),
}

# --- ACTIVE SYSTEM SELECTION ---
SYSTEM_NAME = "mxene_ohflex"
CFG = CONFIGS[SYSTEM_NAME]

# Blocks to analyze. Names must match the density_profile_Mx.in outputs.
# For non-MXene systems (no {block} in path) use BLOCKS = [""].
BLOCKS = ["block1", "block2", "block3", "full"]


@dataclass
class AlgorithmSettings:
    """
    Centralized hyperparameters for physics and math algorithms.
    MODIFY THESE VALUES to tune the contact angle fitting behavior.
    """
    top_layer_tol: float = 0.5       # Tolerance (A) to find substrate top Z
    liquid_cutoff_frac: float = 0.10 # Fraction of max density = 'liquid phase'
    interface_frac: float = 0.45     # Fraction of bulk density = Gibbs surface
    smooth_sigma: float = 1.5        # Gaussian blur sigma for 2D smoothing

    # Exclude the first N Angstrom above the substrate from the BULK estimate
    # (not from the contour). Hydrophilic MXene shows strong layering with
    # peaks up to ~2 g/cm^3 that would bias median/cutoff. Harmless for
    # graphene-type systems (weaker layering).
    bulk_exclude_z: float = 6.0

    # --- FIT WINDOW PARAMETERS (CRITICAL FOR TANGENT PLACEMENT) ---
    # Vertical slice (A above contact point) where the tangent is fitted.
    # Raise 'lo' if the tangent catches the hydration layer,
    # lower 'hi' if it catches the droplet apex.
    z_win_lo_flat: float = 6
    z_win_hi_flat: float = 16
    z_win_lo_pillars: float = 6.0
    z_win_hi_pillars: float = 18.0

    z_bin: float = 0.5               # Bin size for vertical profile compression
    min_pts: int = 5                 # Min points for a valid linear fit

    # --- ROBUST FITTING PARAMETERS ---
    min_r2_target: float = 0.92
    sigma_clip: float = 1.3
    max_clip_iter: int = 5
    window_step: float = 1.5
    max_win_shifts: int = 6

PARAMS = AlgorithmSettings()

PALETTE = {
    "left_tangent": "#C0392B",
    "right_tangent": "#2471A3",
    "contour": "#1C2833",
    "liquid": "#AED6F1",
    "liquid_hist": "#2471A3",
    "bulk_ref": "#922B21",
    "gibbs_ref": "#1A5276",
    "substrate": "#7F8C8D"
}


# =============================================================================
# 2. FILE PARSERS
# =============================================================================

class DataParser:
    """Handles data extraction from LAMMPS output files."""

    @staticmethod
    def get_surface_z(data_file_path: str, substrate_types: Tuple[int, ...],
                      tol: float) -> float:
        """Finds the maximum Z-coordinate across all substrate atom types."""
        z_coords = []
        with open(data_file_path, 'r') as f:
            in_atoms = False
            for line in f:
                line = line.strip()
                if line.startswith("Atoms"):
                    in_atoms = True
                    continue
                if not in_atoms or not line or line.startswith("#"):
                    continue
                if any(line.startswith(s) for s in ("Velocities", "Bonds", "Angles")):
                    break
                parts = line.split()
                if len(parts) >= 7 and int(parts[2]) in substrate_types:
                    z_coords.append(float(parts[6]))

        if not z_coords:
            raise ValueError(
                f"No substrate atoms (types {substrate_types}) in {data_file_path}")

        z_coords = np.array(z_coords)
        return float(np.median(z_coords[z_coords >= z_coords.max() - tol]))

    @staticmethod
    def load_density_dat(dat_file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads 2D chunk/ave density data generated by LAMMPS."""
        rows = []
        with open(dat_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 5:
                    try:
                        rows.append([float(v) for v in parts])
                    except ValueError:
                        continue
        if not rows:
            raise ValueError(f"No valid data rows found in {dat_file_path}")
        arr = np.array(rows)
        return arr[:, 1], arr[:, 2], arr[:, 4]  # x, z, density


# =============================================================================
# 3. DENSITY FIELD PROCESSING
# =============================================================================

class DensityProcessor:
    """Converts 1D chunk data to 2D meshes and calculates droplet properties."""

    @staticmethod
    def estimate_bulk_density(density: np.ndarray, z: np.ndarray,
                              surface_z: float) -> Tuple[float, float]:
        """
        Estimates liquid bulk density. The first bulk_exclude_z A above the
        substrate are excluded: adsorption layering there (rho up to ~2 on
        hydrophilic MXene) would bias both the cutoff and the median.
        """
        above = (z > surface_z + PARAMS.bulk_exclude_z) & (density > 0.0)
        rho_above = density[above]
        if rho_above.size == 0:
            raise ValueError("No density found above the substrate + exclusion zone.")

        cutoff = PARAMS.liquid_cutoff_frac * rho_above.max()
        liquid_bins = rho_above[rho_above > cutoff]
        if liquid_bins.size == 0:
            raise ValueError("No liquid bins found above cutoff.")

        rho_bulk = float(np.median(liquid_bins))
        return rho_bulk, cutoff

    @staticmethod
    def create_grid(x: np.ndarray, z: np.ndarray, density: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Creates a 2D mesh grid for contouring and plotting."""
        xu, zu = np.unique(x), np.unique(z)
        xi = {v: i for i, v in enumerate(xu)}
        zi = {v: i for i, v in enumerate(zu)}

        rho_grid = np.zeros((zu.size, xu.size), dtype=float)
        for xv, zv, rv in zip(x, z, density):
            rho_grid[zi[zv], xi[xv]] = rv
        return xu, zu, rho_grid

    @staticmethod
    def extract_contour(x_grid: np.ndarray, z_grid: np.ndarray, rho_grid: np.ndarray,
                        level: float, surface_z: float) -> Tuple[np.ndarray, np.ndarray]:
        """Finds the Gibbs dividing surface contour from the density field."""
        cut_idx = np.searchsorted(z_grid, surface_z, side="right")
        safe_grid = rho_grid.copy()
        safe_grid[:cut_idx, :] = 0.0

        smooth = gaussian_filter(safe_grid, sigma=PARAMS.smooth_sigma)
        contours = find_contours(smooth, level=level)
        if not contours:
            raise ValueError(f"No contour found at density level {level:.4f}")

        best_contour = None
        best_score = -np.inf
        for c in contours:
            zp = np.interp(c[:, 0], np.arange(z_grid.size), z_grid)
            xp = np.interp(c[:, 1], np.arange(x_grid.size), x_grid)
            score = (xp.max() - xp.min()) * (zp.max() - zp.min())
            if score > best_score:
                best_score = score
                best_contour = (xp, zp)

        return best_contour


# =============================================================================
# 4. MATH LOGIC: TANGENTS & CHECKS
# =============================================================================

class TangentFitter:
    """Primary method: Local tangent calculation with robust sigma-clipping."""

    @staticmethod
    def _compress_side(xs: np.ndarray, zs: np.ndarray, side: str
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Bins data vertically, keeps extreme X per bin (outer boundary only)."""
        z0 = zs.min()
        bidx = np.floor((zs - z0) / PARAMS.z_bin).astype(int)
        xs_out, zs_out = [], []
        for b in np.unique(bidx):
            m = bidx == b
            xs_out.append(xs[m].min() if side == "left" else xs[m].max())
            zs_out.append(zs[m].mean())
        return np.array(xs_out), np.array(zs_out)

    @staticmethod
    def _polyfit_r2(xs: np.ndarray, zs: np.ndarray) -> Tuple[float, float, float]:
        """Linear regression x(z) with R-squared evaluation."""
        m, b = np.polyfit(zs, xs, 1)
        res = xs - (m * zs + b)
        ss_res = np.sum(res**2)
        ss_tot = np.sum((xs - xs.mean())**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return m, b, r2

    @staticmethod
    def _sigma_clip_fit(xs: np.ndarray, zs: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """Iteratively removes outliers to find the macroscopic tangent."""
        m, b, r2 = TangentFitter._polyfit_r2(xs, zs)
        mask = np.ones(len(xs), dtype=bool)

        for _ in range(PARAMS.max_clip_iter):
            residuals = xs - (m * zs + b)
            sigma = residuals.std()
            if sigma < 1e-9:
                break
            new_mask = np.abs(residuals) < PARAMS.sigma_clip * sigma
            if new_mask.sum() < PARAMS.min_pts:
                break
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask
            m, b, r2 = TangentFitter._polyfit_r2(xs[mask], zs[mask])

        return xs[mask], zs[mask], m, b, r2

    @staticmethod
    def fit_side(x_cnt: np.ndarray, z_cnt: np.ndarray, z_contact: float,
                 side: str, is_flat: bool) -> Optional[dict]:
        """Evaluates the contact angle for one side of the droplet."""
        x_mid = np.median(x_cnt)

        z_win_lo = PARAMS.z_win_lo_flat if is_flat else PARAMS.z_win_lo_pillars
        z_win_hi = PARAMS.z_win_hi_flat if is_flat else PARAMS.z_win_hi_pillars
        if CFG.z_win is not None:
            z_win_lo, z_win_hi = CFG.z_win

        n_shifts = CFG.n_win_shifts if CFG.n_win_shifts else \
                   (PARAMS.max_win_shifts if is_flat else 1)

        z_lo0 = z_contact + z_win_lo
        z_hi0 = z_contact + z_win_hi

        best_result = None

        side_mask = (x_cnt < x_mid) if side == "left" else (x_cnt > x_mid)
        xs_s, zs_s = x_cnt[side_mask], z_cnt[side_mask]
        belly_z = float(zs_s[np.argmin(xs_s)]) if side == "left" \
                  else float(zs_s[np.argmax(xs_s)])
        if belly_z > z_contact + 3.0:      # theta > 90: belly above the base
            z_hi0 = min(z_hi0, belly_z - 1.0)

        for shift in range(n_shifts):
            dz = shift * PARAMS.window_step
            z_lo, z_hi = z_lo0 + dz, z_hi0 + dz

            mask = (z_cnt >= z_lo) & (z_cnt <= z_hi)
            mask &= (x_cnt < x_mid) if side == "left" else (x_cnt > x_mid)
            xs_raw, zs_raw = x_cnt[mask], z_cnt[mask]

            if xs_raw.size < PARAMS.min_pts:
                continue

            xs_c, zs_c = TangentFitter._compress_side(xs_raw, zs_raw, side)
            if xs_c.size < PARAMS.min_pts:
                continue

            if is_flat:
                xs_f, zs_f, m, b, r2 = TangentFitter._sigma_clip_fit(xs_c, zs_c)
            else:
                m, b, r2 = TangentFitter._polyfit_r2(xs_c, zs_c)
                xs_f, zs_f = xs_c, zs_c

            angle = float(np.degrees(np.arctan2(1.0, abs(m))))
            if side == "left" and m < 0: angle = 180.0 - angle
            if side == "right" and m > 0: angle = 180.0 - angle

            x_contact = float(m * z_contact + b)

            result = {
                "xs": xs_f, "zs": zs_f, "m": float(m), "b": float(b), "r2": float(r2),
                "angle": angle, "x_contact": x_contact, "side": side,
                "z_win": (z_lo, z_hi)
            }

            if not is_flat:
                return result

            if best_result is None or r2 > best_result["r2"]:
                best_result = result
            if r2 >= PARAMS.min_r2_target:
                break

        return best_result


class SecondaryMath:
    """Alternative measurement heuristics (Geometric evaluation)."""

    @staticmethod
    def geometric_angle(lin_left: dict, lin_right: dict,
                        z_contact: float, z_apex: float) -> Optional[float]:
        """theta = 2 * arctan(h/a), base width from tangent contact points."""
        if not lin_left or not lin_right:
            return None
        a = 0.5 * (lin_right["x_contact"] - lin_left["x_contact"])
        h = z_apex - z_contact
        if a <= 0 or h <= 0:
            return None
        return float(np.degrees(2.0 * np.arctan(h / a)))


# =============================================================================
# 5. VISUALIZATION
# =============================================================================

class Plotter:
    @staticmethod
    def _setup_style():
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.30,
            "grid.linestyle": ":",
            "legend.frameon": True,
            "figure.dpi": 300
        })

    @staticmethod
    def plot_histogram(density, z, surface_z, rho_bulk, cutoff, rho_thr, out_path):
        Plotter._setup_style()
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        fig.patch.set_facecolor("white"); ax.set_facecolor("white")

        above = (z > surface_z + PARAMS.bulk_exclude_z) & (density > 0.0)
        rho_a = density[above]

        ax.hist(rho_a[rho_a > cutoff], bins=200, color=PALETTE["liquid_hist"],
                alpha=0.85, linewidth=0,
                label=f"Liquid phase ($\\rho > {cutoff:.2f}$)")

        ax.axvline(rho_bulk, color=PALETTE["bulk_ref"], lw=1.6, zorder=5,
                   label=f"$\\rho_{{\\rm bulk}} = {rho_bulk:.2f}$")
        ax.axvline(rho_thr, color=PALETTE["gibbs_ref"], lw=1.6, ls="--", zorder=5,
                   label="Gibbs Threshold")
        ax.axvspan(0, cutoff, color="#D5E8F7", alpha=0.45, zorder=0,
                   label="Vapor / Interface")

        ax.set_xlim(0, min(rho_a.max() * 1.05, 1.85))
        ax.set_xlabel("Density, $\\rho$ (g cm$^{-3}$)", fontsize=12)
        ax.set_ylabel("Bin count", fontsize=12)
        ax.legend(fontsize=9, loc="upper right")

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_main(x_grid, z_grid, rho_grid, x_cnt, z_cnt,
                  lin_left, lin_right, theta_geom,
                  z_contact, z_apex, avg_tangent, angle_err,
                  surface_z, out_path, sys_label):
        Plotter._setup_style()
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        fig.patch.set_facecolor("white"); ax.set_facecolor("white")

        ax.set_aspect("equal")

        masked_density = np.ma.masked_where(rho_grid < 0.05, rho_grid)
        levels = np.linspace(0.05, np.max(rho_grid), 15)
        cf = ax.contourf(x_grid, z_grid, masked_density, levels=levels,
                         cmap="Blues", extend="max", alpha=0.85)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.15)
        cbar = fig.colorbar(cf, cax=cax)
        cbar.set_label("Density (g/cm$^3$)", rotation=270, labelpad=15)

        ax.plot(x_cnt, z_cnt, "-", color=PALETTE["contour"], lw=1.8,
                label="Gibbs dividing surface", zorder=3)
        ax.axhline(surface_z, color=PALETTE["substrate"], ls=":", lw=1.5,
                   zorder=2, label="Substrate plane")

        z_line = np.array([surface_z - 0.5, z_contact + PARAMS.z_win_hi_pillars + 3])
        for fit, color, label_text in [(lin_left, PALETTE["left_tangent"], "Left tangent"),
                                       (lin_right, PALETTE["right_tangent"], "Right tangent")]:
            if not fit:
                continue
            x_full = fit["m"] * z_line + fit["b"]
            ax.plot(x_full, z_line, "--", color=color, lw=1.8, zorder=5,
                    solid_capstyle="round", label=label_text)
            ax.plot(fit["x_contact"], surface_z, "o", ms=6, mfc="white",
                    mec=color, mew=1.6, zorder=8)

            t1, t2 = (90.0, 90.0 + (180.0 - fit["angle"])) if fit["side"] == "left" \
                     else (90.0 - (180.0 - fit["angle"]), 90.0)
            ax.add_patch(mpatches.Arc((fit["x_contact"], surface_z),
                                      width=12, height=12,
                                      theta1=t1, theta2=t2, color=color,
                                      lw=1.4, zorder=9))

        ax.set_xlabel("$x$ (Å)", fontsize=12)
        ax.set_ylabel("$z$ (Å)", fontsize=12)
        ax.set_xlim(x_cnt.min() - 15, x_cnt.max() + 15)
        ax.set_ylim(surface_z - 5, z_apex + 30)

        angle_text = rf"$\theta = {avg_tangent:.1f}^\circ \pm {angle_err:.1f}^\circ$"
        ax.text(0.50, 0.73, angle_text, transform=ax.transAxes, fontsize=14,
                va='top', ha='center', color='black')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=10,
                  loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=False)

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()


# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

def analyze(block: str) -> Optional[dict]:
    """Full analysis pipeline for one density file (one block or full)."""
    density_path = CFG.density_file.format(block=block) if "{block}" in CFG.density_file \
                   else CFG.density_file
    if not Path(density_path).exists():
        print(f"[skip] {density_path} not found")
        return None

    tag = f"{SYSTEM_NAME}_{block}" if block else SYSTEM_NAME
    print(f"\n--- [{tag}] ---")

    surface_z = DataParser.get_surface_z(CFG.data_file, CFG.substrate_types,
                                         PARAMS.top_layer_tol)
    x, z, density = DataParser.load_density_dat(density_path)
    print(f"Substrate Z: {surface_z:.2f} Å")

    rho_bulk, cutoff = DensityProcessor.estimate_bulk_density(density, z, surface_z)
    rho_thr = rho_bulk * PARAMS.interface_frac
    print(f"Bulk density: {rho_bulk:.3f} g/cm³, Gibbs threshold: {rho_thr:.3f} g/cm³")

    x_grid, z_grid, rho_grid = DensityProcessor.create_grid(x, z, density)
    x_cnt, z_cnt = DensityProcessor.extract_contour(x_grid, z_grid, rho_grid,
                                                    rho_thr, surface_z)
    z_contact, z_apex = float(z_cnt.min()), float(z_cnt.max())

    lin_left = TangentFitter.fit_side(x_cnt, z_cnt, z_contact, "left", CFG.is_flat)
    lin_right = TangentFitter.fit_side(x_cnt, z_cnt, z_contact, "right", CFG.is_flat)

    if not lin_left or not lin_right:
        print(f"[{tag}] ERROR: tangent fitting failed.")
        return None

    avg_tangent = 0.5 * (lin_left["angle"] + lin_right["angle"])
    angle_err = abs(lin_left["angle"] - lin_right["angle"]) / 2.0
    theta_geom = SecondaryMath.geometric_angle(lin_left, lin_right, z_contact, z_apex)

    print(f"Tangents -> Left: {lin_left['angle']:.1f}° (R²={lin_left['r2']:.3f}) | "
          f"Right: {lin_right['angle']:.1f}° (R²={lin_right['r2']:.3f})")
    print(f"RESULT: {avg_tangent:.2f}° ± {angle_err:.2f}°"
          + (f" | geometric check: {theta_geom:.2f}°" if theta_geom else ""))

    out_dir = Path(CFG.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Plotter.plot_histogram(density, z, surface_z, rho_bulk, cutoff, rho_thr,
                           out_dir / f"histogram_{tag}.png")
    Plotter.plot_main(x_grid, z_grid, rho_grid, x_cnt, z_cnt,
                      lin_left, lin_right, theta_geom,
                      z_contact, z_apex, avg_tangent, angle_err,
                      surface_z, out_dir / f"contact_angle_{tag}.png", CFG.name)

    return {"block": block or "single", "theta": avg_tangent, "err": angle_err,
            "left": lin_left["angle"], "right": lin_right["angle"],
            "geom": theta_geom, "rho_bulk": rho_bulk,
            "r2_l": lin_left["r2"], "r2_r": lin_right["r2"]}


def main():
    print(f"\n[{CFG.name}] Contact angle analysis, blocks: {BLOCKS}")
    results = []
    for b in BLOCKS:
        r = analyze(b)
        if r:
            results.append(r)

    if not results:
        print("\nNo results produced.")
        return

    header = (f"{'block':<10}{'theta':>8}{'+-':>6}{'left':>8}{'right':>8}"
              f"{'geom':>8}{'rho_b':>8}{'R2_L':>7}{'R2_R':>7}")
    lines = [header]
    print("\n" + header)
    for r in results:
        line = (f"{r['block']:<10}{r['theta']:>8.1f}{r['err']:>6.1f}"
                f"{r['left']:>8.1f}{r['right']:>8.1f}"
                f"{(r['geom'] if r['geom'] else float('nan')):>8.1f}"
                f"{r['rho_bulk']:>8.3f}{r['r2_l']:>7.3f}{r['r2_r']:>7.3f}")
        print(line)
        lines.append(line)

    summary = Path(CFG.out_dir) / f"summary_{SYSTEM_NAME}.txt"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines) + "\n")
    print(f"\nSummary -> {summary}")
    print(f"Plots   -> {CFG.out_dir}")

if __name__ == "__main__":
    main()