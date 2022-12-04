# -*- coding: utf-8 -*-
"""
MiePython simulation 
JFL
"""
import numpy as np
import matplotlib.pyplot as plt
import miepython

# Set the optical properties of Au (fromb https://refractiveindex.info/?shelf=main&book=Au&page=Johnson)
# Wavelength is given in µm
ref_index=np.loadtxt('C:\\Users\\haq\\Desktop\\comp project\\n k 2b.txt')

#Few lines to separate the columns
lam = ref_index[:,0] #Wavelength
n = ref_index[:,1] #real part
k = ref_index[:,2] #imaginary part

# Define few parameters for your simulation
radius = (0.007) # NPs diameter in µm
n_env= 1.33 # Refractive index of medium surronding the NPs

# Necessary to arrange the data a little bit for the simulation 
m = n - 1.0j * k
x = 2 * np.pi * radius / lam
cross_section_area = np.pi * radius ** 2
mu_a = 4 * np.pi * k / lam    
mm = m/n_env
xx = 2*np.pi*radius*n_env/lam

# Do the calculation using the MiePython simulation
qext, qscat, qback, g = miepython.mie(mm, xx)

# Collect the results 
sca_cross_section = qscat * cross_section_area
abs_cross_section = (qext - qscat) * cross_section_area
ext_cross_section = qext * cross_section_area

# You can also use python for displaying all the results !
    # Plot the refractive index you collected from the database
plt.figure()
plt.scatter(lam,n,color='blue')
plt.scatter(lam,k,color='red')
#plt.xlim((0.2,2))
plt.xlabel('Wavelength (microns)')
plt.ylabel('Refractive Index')
plt.annotate(r'$m_\mathrm{re}$', xy=(1.0,0.5),color='blue')
plt.annotate(r'$m_\mathrm{im}$', xy=(1.0,8),color='red')
plt.title('Complex Refractive Index of Gold')
plt.tight_layout()
#plt.savefig("plot_1.png")
plt.show()

    # Plot the NP spectrum (abs/scat/ext)
plt.figure()
plt.plot()
plt.plot(lam * 1000, abs_cross_section, color='blue')
plt.plot(lam * 1000, sca_cross_section, color='red')
plt.plot(lam * 1000, ext_cross_section, color='green')
plt.title ('Au NP (r=40nm, n=1.33)')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Cross Section (µm²)")
#plt.xlim(400, 800) # To be limited to the visible
#plt.text(700, 0.01, "Abs", color='blue')
#plt.text(750, 0.08, "Scat", color='red')
#plt.text(750, 0.1, "Ext", color='green')
#plt.savefig("plot_2.png")
plt.show()

# To get nicer results to plot because there is not enough points for the curve (only ext in this case)
plt.figure()
from scipy.interpolate import interp1d
new = np.linspace(lam.min(),lam.max(), 1000)
smooth= interp1d(lam,ext_cross_section, kind='cubic')
plt.plot(new*1000,smooth(new))
plt.xlabel("Wavelength (nm)")
plt.ylabel("Cross Section (µm²)")
plt.xlim(300,1000)
plt.show()

