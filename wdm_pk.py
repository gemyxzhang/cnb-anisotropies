import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from classy import Class
from scipy.optimize import fsolve
from scipy import special # bessel functions 
from matplotlib import gridspec
from matplotlib.legend import Legend
import time  # temp timer 
import cnb_utils as uts

# constants 
h = 0.67
c = 299793.
H0 = 100*h
omega_m = 0.27 + 0.049
omega_rad = 2.47e-05/(h*h)
omega_lambda = 1 - omega_m - omega_rad

T_nu = 2.7255*(4/11)**(1/3)*(3.046/3)**(1/4)
k_B = 8.617333e-05 # eV/K
a_ndec = 1e-10

massless_cutoff = 2e-5   # below this, we consider neutrinos to be massless
scaling = 1e12  # for scaling cl to the correct unit/magnitude 

colors = ["#C74804", "#66bda5", "#CC78BC", "#DE8F05", "#0173B2"]

# changing variables 
q_i = 1  # placeholder q index 0=1.5 (max q/n_qbins),1=3.0,2=4.5,3=6.0,4=7.5,5=9.0,6=10.5,7=12.0,8=13.5,9=15.0

# relevant values for running the functions 
l_min = 1
l_max = 5
ls = np.arange(l_min, l_max+1, 1)

nu_masses_str = ['0.00001','0.01', '0.05', '2000'] 
m1, m2, m3, m4 = float(nu_masses_str[0]), float(nu_masses_str[1]), float(nu_masses_str[2]), float(nu_masses_str[3])
nu_masses = [m1, m2, m3, m4]  # eV
k_min, k_max = 1e-2, 1e1    # 1/Mpc
n_kmodes = 100
k_magnitudes_full = np.logspace(np.log(k_min), np.log(k_max), n_kmodes, base=np.e)
l_max_ncdm = 50
n_qbins = 10
uts.n_qbins = n_qbins

As_default = 2.215e-9
pivot_default = 5e-2
pivot = 5e-4 
ns = 0.9619
As = As_default*(pivot/pivot_default)**(ns-1)

ofac = 0.010736525660298
zs = [0, 0.8, 1.0, 1.2]  # redshifts for class to output pks 

params = {
    # Output
    'output'         : 'mPk',
    'k_output_values': list(k_magnitudes_full),
    'P_k_max_1/Mpc'  : 0,
    # Basic cosmology
    'H0'       : 67.11,
    'omega_b'  : 0.022068,
    'omega_cdm': 0.12, 
    'omega_ncdm': [m1*ofac, m2*ofac, m3*ofac],
    'Omega_Lambda': 0.6825,
    'Omega_k' : 0.,
    'n_s': 0.9624,
    # Add neutrino hierarchy
    'N_ur'    : 0,
    'N_ncdm'  : 3,
    'm_ncdm'  : nu_masses[:-1],
    'l_max_ncdm'              : l_max_ncdm,
    # Photon temperature and precision parameters
    'T_cmb'                            : 2.726,
    'evolver'                     : 0,
    'gauge':'synchronous', 
    'z_pk': zs
}

params2 = {
    # Output
    'output'         : 'mPk',
    'k_output_values': list(k_magnitudes_full),
    'P_k_max_1/Mpc'  : 0,
    # Basic cosmology
    'H0'       : 67.11,
    'omega_b'  : 0.022068,
    'omega_cdm': 0.000000001,
    'omega_ncdm': [m1*ofac, m2*ofac, m3*ofac, 0.12],
    'Omega_Lambda': 0.6825,
    'Omega_k' : 0.,
    'n_s': 0.9624,
    # Add neutrino hierarchy
    'N_ur'    : 0,
    'N_ncdm'  : 4,
    'm_ncdm'  : nu_masses,
    'l_max_ncdm'              : l_max_ncdm,
    # Photon temperature and precision parameters
    'T_cmb'                            : 2.726,
    'evolver'                     : 0,
    'gauge':'synchronous',
    'z_pk': zs
}

for key, val in params.copy().items():
    if isinstance(val, list):
        params[key] = str(val).strip('[]')
        
for key, val in params2.copy().items():
    if isinstance(val, list):
        params2[key] = str(val).strip('[]')
        
# run CLASS with CDM and WDM 
start_time = time.time()

cosmo = Class()
cosmo.set(params)
cosmo.compute()

print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()

cosmo2 = Class()
cosmo2.set(params2)
cosmo2.compute()

print("--- %s seconds ---" % (time.time() - start_time))

pks = []
pks2 = []

# get the pk at each redshift 
for j in range(len(zs)): 
    pk, pk2 = [], [] 
    for ki in k_magnitudes_full[:-1]: 
        pk.append(cosmo.pk(ki, zs[j])) 
        pk2.append(cosmo2.pk(ki, zs[j]))
        
    pks.append(pk)
    pks2.append(pk2)
    
os.makedirs('./arrays/', exist_ok=True)
# save arrays 
np.savez('./arrays/cdmpk_k{}to{}'.format(k_min, k_max), ks=k_magnitudes_full, zs=zs, pks=pks)    
np.savez('./arrays/wdmpk_k{}to{}'.format(k_min, k_max), ks=k_magnitudes_full, zs=zs, pks=pks2)  

f, ax = plt.subplots(1, 1, figsize=(6, 6))
plots = []

# plot the pks for WDM and CDM for each redshift 
for j in range(len(zs)): 
    if (j == 0): 
        plots += ax.loglog(k_magnitudes_full[:-1], pks[j], label=r'$z = {}$'.format(zs[j]), color=colors[j])
        plots += ax.loglog(k_magnitudes_full[:-1], pks2[j], color=colors[j], linestyle='--')
    else: 
        ax.loglog(k_magnitudes_full[:-1], pks[j], label=r'$z = {}$'.format(zs[j]), color=colors[j])
        ax.loglog(k_magnitudes_full[:-1], pks2[j], color=colors[j], linestyle='--')

ax.set_xlabel(r'$k$ (1/Mpc)')
ax.set_ylabel(r'P$(k)$')
ax.legend()

leg = Legend(ax, plots, ['CDM', 'WDM'], frameon=False)
ax.add_artist(leg) 

plt.savefig('pk.pdf')