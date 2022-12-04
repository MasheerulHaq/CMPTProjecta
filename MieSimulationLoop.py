# -*- coding: utf-8 -*-
"""
MiePython simulation 
JFL
"""
import numpy as np
import matplotlib.pyplot as plt
import miepython
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
#from scipy.optimize import leastsq
from scipy.optimize import curve_fit
#%%
# Set the optical properties of Au (fromb https://refractiveindex.info/?shelf=main&book=Au&page=Johnson)
# Wavelength is given in µm
ref_index=np.loadtxt('C:\\Users\\haq\\Desktop\\comp project\\n k 2b.txt') # Change path for your own computer

#Few lines to separate the columns
lam = ref_index[:,0] #Wavelength
n = ref_index[:,1] #real part
k = ref_index[:,2] #imaginary part
new = np.linspace(lam.min(),lam.max(), 1000)

# Define few parameters for your simulation
radius = 0.03 # NPs diameter in µm
#n_env= 1.33 # Refractive index of medium surronding the NPs

n_envs= np.linspace(1, 2, 30)
results=[]

#%%
#Run calculation

for i in n_envs:
    n_env=i

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
    results.append(ext_cross_section)
    

# You can also use python for displaying all the results !
    # Plot the refractive index you collected from the database

#%%
# to display the spectra

for i in results:
#    plt.figure()
    smooth= interp1d(lam,i, kind='cubic')
    plt.plot(new*1000,smooth(new), color='blue')
    plt.xlim((400,900))
    plt.title ('effect of changing surrounding refractive index')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Extinction Cross section (µm)2')
    plt.annotate(r'$m_\mathrm{re}$', xy=(1.0,0.5),color='blue')
    plt.annotate(r'$m_\mathrm{im}$', xy=(1.0,8),color='red')
   
    

#    plt.title('Complex Refractive Index of Gold')
    plt.tight_layout()
#plt.savefig("plot_1.png")
plt.show()

#%%
    
#Compute plasmon sensitivity  
lambdamax=[]
for i in results:
    smooth= interp1d(lam,i, kind='cubic')    
    max_x = new[smooth(new).argmax()]  # Find the x value corresponding to the maximum y value
    lambdamax.append(max_x)
    

#%%

#Display results of sensitivity
# extract intercept b and slope m
lambdamax_nm=np.array(lambdamax)*1000
reg = LinearRegression().fit(n_envs[:, None], lambdamax_nm)
b = reg.intercept_
m = reg.coef_[0]
plt.figure()
plt.axline(xy1=(0, b), slope=m, label=f'$y = {m:.1f}x {b:+.1f}$')
plt.scatter(n_envs,lambdamax_nm)
plt.xlim(1, 1.6)
#plt.ylim(0.5,0.65)
plt.xlabel('refractive index')
plt.ylabel('plasmon shift (nm)')
plt.show()
print('Sensitivity equal to',m, 'nm/RIU')

#%%

#fitting

def gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

x=np.array(new)*1000
y=np.array(smooth(new))

n = len(x)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y*(x-mean)**2)/n)

popt,pcov = curve_fit(gauss,x,y,p0=[max(y),mean,sigma])

plt.plot(x,y,'b+:',label='data')
plt.plot(x,gauss(x,*popt),'ro:',label='fit')
plt.legend()
plt.title('Fitting')
plt.xlabel('wavelength (nm)')
plt.ylabel('Ex cross section (µm)2')
plt.show()

print('FWHM',popt[2]*2.354)
print('Figure of Merit', m/(popt[2]*2.354))


