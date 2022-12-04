# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 22:31:58 2022

@author: haq
"""

## --- load the modules
import numpy as np
import matplotlib.pyplot as plt
from pyGDM2 import structures
from pyGDM2 import materials
from pyGDM2 import fields
from pyGDM2 import core
from pyGDM2 import propagators
from pyGDM2 import visu
from pyGDM2 import tools
from pyGDM2 import linear

## --- simulation initialization ---
step = 10
material = materials.gold() # For using gold as materials
geometry = structures.rect_wire(step, L=3, W=3, H=3) # Change the NP dimension and geometry

struct = structures.struct(step, geometry, material)
field_generator = fields.plane_wave
wavelengths = np.linspace(300, 1200, 31)
kwargs = dict(theta=[0,90], inc_angle=180)
efield = fields.efield(field_generator, wavelengths=wavelengths, kwargs=kwargs)
dyads = propagators.DyadsQuasistatic123(n1=1)
sim_polarizations = core.simulation(struct, efield, dyads)

## --- run the simulation ---
sim_polarizations.scatter()
    #Get the two polarization direction
wl, spec0 = tools.calculate_spectrum(sim_polarizations, 0, linear.extinct)
ex_p, sc, ab = spec0.T
wl, spec1 = tools.calculate_spectrum(sim_polarizations, 1, linear.extinct)
ex_s, sc, ab = spec1.T
ex=(ex_p+ex_s)/2

## --- plot results ---
    # NP geometry
visu.structure(sim_polarizations)
print("N dp={}".format(len(geometry)))
    # Spectrum
plt.plot(wl, ex, label='total extinction')
plt.legend()
plt.title("multipole decomposition - extinction")
plt.xlabel("wavelength (nm)")
plt.ylabel("extinction cross section (nm^2)")
plt.show()