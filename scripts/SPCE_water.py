#!/usr/bin/env python3
"""
Generates Packmol input for an SPC/Fw water droplet (R=40A) and runs Packmol
to create the water droplet coordinates. Compatible with flexible model.

Requires Packmol installed and a correct water.xyz file with SPC/Fw geometry.
"""

import subprocess
import math
import sys
import textwrap
import os
import random

# --- Parameters ---
droplet_radius = 40.0
gap = 4.0  # Initial gap to graphene
graphene_z = 0.0
packmol_tolerance = 1.5
seed = random.randint(1, 100000)

# File paths
water_template_file = "/mnt/c/Users/vbarv/Desktop/course/project/lammps/data/water.xyz"
packmol_output_file = "/mnt/c/Users/vbarv/Desktop/course/project/lammps/data/water_droplet_packed.xyz"
packmol_inp_file = "packmol_auto.inp"
packmol_log_file = "packmol_auto.log"

# Approximate volume per molecule
water_mol_volume = 30.0
volume = (4.0 / 3.0) * math.pi * droplet_radius**3
Nwater = int(volume / water_mol_volume)
print(f"Targeting {Nwater} water molecules for R={droplet_radius} Å")

center_z = graphene_z + droplet_radius + gap

packmol_input = f"""
tolerance {packmol_tolerance}
filetype xyz
output {packmol_output_file}
seed {seed}

structure {water_template_file}
  number {Nwater}
  inside sphere 0.0 0.0 {center_z:.2f} {droplet_radius}
end structure
"""
with open(packmol_inp_file, "w") as f:
    f.write(textwrap.dedent(packmol_input))

# --- Run Packmol ---
print(f"Running Packmol, output to {packmol_log_file}...")
cmd = f"packmol < {packmol_inp_file} > {packmol_log_file} 2>&1"

try:
    subprocess.run(cmd, shell=True, check=True, timeout=900)

    if not os.path.exists(packmol_output_file) or os.path.getsize(packmol_output_file) < 50:
        print(f"Warning: Output file seems empty: {packmol_output_file}")
        with open(packmol_log_file) as logf:
            print(logf.read())
except Exception as e:
    print(f"Packmol error: {e}")
    if os.path.exists(packmol_log_file):
        with open(packmol_log_file) as logf:
            print(logf.read())
    sys.exit(1)

print(f"\n✅ Water droplet generated: {packmol_output_file}")
