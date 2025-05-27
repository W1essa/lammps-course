#!/usr/bin/env python3
import numpy as np
from ase.build import graphene
from ase.io import write
import os

# === PARAMETERS ===
structure_type = 'pillars'  # 'flat' або 'pillars'
target_L = 200.0         # розмір графену, Å
pillar_period = 20.0     # період наноструктури, Å
pillar_radius = 8.0     # радіус стовпчика, Å
pillar_height = 10.0     # висота стовпчика, Å
c_c_bond = 1.42          # довжина C–C, Å

output_dir = "/mnt/c/Users/vbarv/Desktop/course/project/lammps/data/"
output_file = os.path.join(output_dir, f"graphene_{structure_type}.xyz")
os.makedirs(output_dir, exist_ok=True)

# === Генеруємо плоский графен через ASE ===
a_graphene = c_c_bond * np.sqrt(3)
graphene_sheet = graphene(size=(int(target_L/a_graphene), int(target_L/a_graphene), 1),
                          vacuum=10.0)
graphene_sheet.center(axis=(0, 1), vacuum=0.0)
positions = graphene_sheet.get_positions()

# === Додаємо наноструктури (плавні пагорби) ===
if structure_type == 'pillars':
    def pillar_height_smooth(x, y):
        cx = (x % pillar_period) - pillar_period/2
        cy = (y % pillar_period) - pillar_period/2
        r = np.sqrt(cx**2 + cy**2)
        if r <= pillar_radius:
            return pillar_height * 0.5 * (1 + np.cos(np.pi * r / pillar_radius))
        return 0.0

    for i, (x,y,z) in enumerate(positions):
        positions[i,2] += pillar_height_smooth(x,y)

graphene_sheet.set_positions(positions)

# === Зберігаємо в XYZ ===
write(output_file, graphene_sheet)
print(f"Graphene '{structure_type}' generated: {len(graphene_sheet)} atoms → {output_file}")
