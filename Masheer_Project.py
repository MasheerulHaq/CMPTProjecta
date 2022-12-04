import numpy as np
import matplotlib.pyplot as plt
import miepython
import csv

ref_index=np.loadtxt("kiana.csv", delimiter=',')
lam = ref_index[:,0]
n = ref_index[:,1]
k = ref_index[:,2]

diameter = 200
print(lam)
pol = ref_index[:,1]
print(pol)
bol = ref_index[:,2]
print(bol)

plt.plot(lam, n)
# comparison betweem lam and n
plt.title("Comparison of extinction for absorbing and non-absorbing spheres")
plt.xlabel("Size Parameter (-)")
plt.ylabel("Qext")
plt.legend()
# plt.show()

print("Please enter the diameter of the particles in nm:")
diameter = int(input())
print("Please select surrounding medium (air, water, acetone, ethanol,toluene, hexane, chloroform, or propanol)")
n_env=input()
if n_env=='air':
    n_env=1.000
elif n_env=='water':
    n_env=1.333
elif n_env=='acetone':
    n_env=1.359
elif n_env=='ethanol':
    n_env=1.36
elif n_env=='toluene':
    n_env=1.496
elif n_env=='hexane':
    n_env=1.375
elif n_env=='chloroform':
    n_env=1.445
elif n_env=='propanol':
    n_env=1.387
else:
    print("Invalid input")

    # If the input parameters are correct, proceed further. Otherwise, theprogram will be terminated.
if isinstance(n_env,float)==True:
# Calculate the scattering (qqsca), absorption (qqabs) and extinction (qqext) efficiencies
    num = len(lam)
    m = (n-1.0j*k)/n_env
    var = np.pi*diameter/lam*n_env
    qqabs = np.zeros(num)
    qqsca = np.zeros(num)
    qqext = np.zeros(num)

for i in range(num) :
    qext, qsca, qback, g = miepython.mie(m[i],var[i])
    qabs = qext - qsca
    qqabs[i]=qabs
    qqsca[i]=qsca
    qqext[i]=qext

fig, ax = plt.subplots()
ax.plot(lam, qqabs, linewidth = '3')
ax.plot(lam, qqsca, linewidth = '3')
ax.plot(lam, qqext, linewidth = '3')
ax.set(xlabel='Wavelength [nm]', ylabel='Efficiency [a.u.]', title='{}'.format(diameter)+' nm Au nanosphere (water)')
ax.legend(['Absorption', 'Scattering', 'Extinction'])
plt.show()

CMF=np.loadtxt("kiana.csv", delimiter=',')

x = CMF[:,0]
y = CMF[:,1]
z = CMF[:,2]
# Load relative spectral power of a standard illuminant D65
D65 = CMF[:,3]
# Load conversion matrix from XYZ coordinates to sRGB
M_sRGB=np.array([[3.2404542, -1.5371385, -0.4985314],[-0.9692660,
1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])
spectra = np.column_stack((qqabs, qqsca, qqext))
rgb = np.zeros((3,3))

for i in range(3):
    spectrum = spectra[:,i]
# Convert the spectrum to CIE chromaticity values (X,Y,Z) by integrating the spectrum over the color matching functions
    D65_x_r_sum=np.sum(x*D65*spectrum)
    D65_y_r_sum=np.sum(y*D65*spectrum)
    D65_z_r_sum=np.sum(z*D65*spectrum)
    D65_x_sum=np.sum(x*D65)
    D65_y_sum=np.sum(y*D65)
    D65_z_sum=np.sum(z*D65)
# Remove the intensity dependence of the calculated color
    X = D65_x_r_sum/D65_y_sum
    Y = D65_y_r_sum/D65_y_sum
    Z = D65_z_r_sum/D65_y_sum
# Calculate normalized chromaticities
    xx = X/(X+Y+Z)
    yy = Y/(X+Y+Z)
    zz = Z/(X+Y+Z)
# Convert to standard linear RGB
    rgb[i,:]=np.dot(M_sRGB,[xx,yy,zz])


# Transform linear RGB values to nonlinear RGB values through gamma correction https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
for i in range(3):
    for j in range(3):
        if rgb[i,j] <= 0.0031308:
            rgb[i,j] = 12.92*rgb[i,j]
        else:
            rgb[i,j] = 1.055*rgb[i,j]**(1/2.4)-0.055
# Clip the values outside of the RGB gamut
        if rgb[i,j] < 0:
            rgb[i,j] = 0
        if rgb[i,j] > 1:
            rgb[i,j] = 1
# Display the RGB values in the 0...255 range.
# For the extinction spectrum, the opposite color is displayed to represent the perceived color
print("Absorbed color: ", np.around(255*rgb[0,:]))
print("Scattered color: ", np.around(255*rgb[1,:]))
print("Transmitted color: ", np.around(255-255*rgb[2,:])) 
#else:
#exit: