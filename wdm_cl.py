import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from classy import Class
from scipy.optimize import fsolve
from scipy import special # bessel functions 
from matplotlib import gridspec
from matplotlib.legend import Legend
import os, time  # temp timer 
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
l_max = 150
ls = np.arange(l_min, l_max+1, 1)

nu_masses_str = ['0.00001','0.01', '0.05', '2000'] 
m1, m2, m3, m4 = float(nu_masses_str[0]), float(nu_masses_str[1]), float(nu_masses_str[2]), float(nu_masses_str[3])
nu_masses = [m1, m2, m3, m4]  # eV
k_min, k_max = 1e0, 1e1    # 1/Mpc
n_kmodes = 10
k_magnitudes_full = np.logspace(np.log(k_min), np.log(k_max), n_kmodes, base=np.e)
l_max_ncdm = l_max
n_qbins = 10
uts.n_qbins = n_qbins

As_default = 2.215e-9
pivot_default = 5e-2
pivot = 5e-4 
ns = 0.9619
As = As_default*(pivot/pivot_default)**(ns-1)

ofac = 0.010736525660298

params = {
    # Output
    'output'         : 'dTk',
    'k_output_values': list(k_magnitudes_full),
    'P_k_max_1/Mpc'  : 0,
    # Basic cosmology
    'H0'       : 67,
    'Omega_b'  : 0.049,
    'omega_cdm': 0.000000001,
    'omega_ncdm': [m1*ofac, m2*ofac, m3*ofac, 0.12],
    'Omega_Lambda': 0.6825,
    'Omega_k' : 0.,
    'n_s': ns,
    # Add neutrino hierarchy
    'N_ur'    : 0,
    'N_ncdm'  : 4,
    'm_ncdm'  : nu_masses,
    # Neutrino precision parameters
    'ncdm_fluid_approximation': 3,
    'Quadrature strategy'     : [3]*4,
    'l_max_ncdm'              : l_max_ncdm,
    'Number of momentum bins' : [n_qbins]*4,
    'Maximum q'               : [15]*4,
    # Photon temperature and precision parameters
    'T_cmb'                            : 2.7255,
    'radiation_streaming_approximation': 3,
    # General precision parameters
    'evolver'                     : 0,
    'recfast_Nz0'                 : 1e+5,
    'tol_thermo_integration'      : 1e-6,
    # add for phi_prime
    'extra metric transfer functions':'y',
    'gauge':'synchronous',
}

params2 = {
    # Output
    'output'         : 'dTk',
    'k_output_values': list(k_magnitudes_full),
    'P_k_max_1/Mpc'  : 0,
    # Basic cosmology
    'H0'       : 67,
    'Omega_b'  : 0.049,
    'omega_cdm': 0.12,
    'omega_ncdm': [m1*ofac, m2*ofac, m3*ofac],
    'Omega_Lambda': 0.6825,
    'Omega_k' : 0.,
    'n_s': ns,
    # Add neutrino hierarchy
    'N_ur'    : 0,
    'N_ncdm'  : 3,
    'm_ncdm'  : nu_masses[:-1],
    # Neutrino precision parameters
    'ncdm_fluid_approximation': 3,
    'Quadrature strategy'     : [3]*3,
    'l_max_ncdm'              : l_max_ncdm,
    'Number of momentum bins' : [n_qbins]*3,
    'Maximum q'               : [15]*3,
    # Photon temperature and precision parameters
    'T_cmb'                            : 2.7255,
    # General precision parameters
    'evolver'                     : 0,
    'recfast_Nz0'                 : 1e+5,
    'tol_thermo_integration'      : 1e-6,
    # add for phi_prime
    'extra metric transfer functions':'y',
    'gauge':'synchronous',
}


for key, val in params.copy().items():
    if isinstance(val, list):
        params[key] = str(val).strip('[]')

for key, val in params2.copy().items():
    if isinstance(val, list):
        params2[key] = str(val).strip('[]')
        
# run CLASS with WDM and CDM separately 
pts_full, tau_0_full = uts.run_class(params, False)
pts_full2, tau_0_full2 = uts.run_class(params2, False)

# get the Cls
clslos0_full = uts.get_clq_LoS(0, q_i, pts_full, None, nu_masses, True, k_magnitudes_full, ls, tau_0_full, np.infty)   
clslos1_full = uts.get_clq_LoS(1, q_i, pts_full, None, nu_masses, True, k_magnitudes_full, ls, tau_0_full, np.infty)
clslos2_full = uts.get_clq_LoS(2, q_i, pts_full, None, nu_masses, True, k_magnitudes_full, ls, tau_0_full, np.infty)

clsbh0_full, clsbh1_full, clsbh2_full = uts.get_clq_BH(q_i, pts_full, k_magnitudes_full, ls, True, ns)

print('', flush=True) 

clslos0_full2 = uts.get_clq_LoS(0, q_i, pts_full2, None, nu_masses, True, k_magnitudes_full, ls, tau_0_full2, np.infty)   
clslos1_full2 = uts.get_clq_LoS(1, q_i, pts_full2, None, nu_masses, True, k_magnitudes_full, ls, tau_0_full2, np.infty)
clslos2_full2 = uts.get_clq_LoS(2, q_i, pts_full2, None, nu_masses, True, k_magnitudes_full, ls, tau_0_full2, np.infty)

clsbh0_full2, clsbh1_full2, clsbh2_full2 = uts.get_clq_BH(q_i, pts_full2, k_magnitudes_full, ls, True, ns)

print('', flush=True) 

# plot the cls 
plt.semilogy(ls, ls*(ls+1)*clslos2_full/(2.0*np.pi), label=r'WDM, m = {} eV'.format(nu_masses[3]))
plt.semilogy(ls, ls*(ls+1)*clslos2_full2/(2.0*np.pi), label='CDM')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)C_\ell/2\pi \ \ [\mu K^2]$')
plt.legend()

plt.savefig('cls.pdf')

# plot the ratio 
f = plt.figure(figsize=(8,8))
plt.plot(ls, clslos2_full/clslos2_full2)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_{\ell, WDM}/C_{\ell, CDM}$')
plt.savefig('cl_ratio.pdf')

os.makedirs('./arrays/', exist_ok=True)

# save all the arrays 
np.savez('./arrays/cdmcls_m{}_lmax{}_k{}to{}'.format(nu_masses[2], l_max, k_min, k_max), cls=clslos2_full2, cls_bh=clsbh2_full2, ks=k_magnitudes_full)    
np.savez('./arrays/wdmcls_m{}_lmax{}_k{}to{}'.format(nu_masses[2], l_max, k_min, k_max), cls=clslos2_full, cls_bh=clsbh2_full, ks=k_magnitudes_full)
np.savez('./arrays/cdmcls_m{}_lmax{}_k{}to{}'.format(nu_masses[0], l_max, k_min, k_max), cls=clslos0_full2, cls_bh=clsbh0_full2, ks=k_magnitudes_full)
np.savez('./arrays/wdmcls_m{}_lmax{}_k{}to{}'.format(nu_masses[0], l_max, k_min, k_max), cls=clslos0_full, cls_bh=clsbh0_full, ks=k_magnitudes_full) 
np.savez('./arrays/cdmcls_m{}_lmax{}_k{}to{}'.format(nu_masses[1], l_max, k_min, k_max), cls=clslos1_full2, cls_bh=clsbh1_full2, ks=k_magnitudes_full)
np.savez('./arrays/wdmcls_m{}_lmax{}_k{}to{}'.format(nu_masses[1], l_max, k_min, k_max), cls=clslos1_full, cls_bh=clsbh1_full, ks=k_magnitudes_full)

# plot for verification 

# adjust row widths of the subplots 
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2])

f = plt.figure(figsize=(16,10))
ax1 = f.add_subplot(gs[0,0])
ax2 = f.add_subplot(gs[1,0], sharex=ax1)
ax3 = f.add_subplot(gs[0,1], sharey=ax1)
ax4 = f.add_subplot(gs[1,1], sharex=ax3, sharey=ax2)

plots = [] # for second legend 

plots += ax1.semilogy(ls, ls*(ls+1)*clslos0_full/(2.0*np.pi), color=colors[0], marker=',', 
                     label=r'$m_\nu = {}\, \mathrm{{eV}}$'.format(nu_masses_str[0]))
ax1.semilogy(ls, ls*(ls+1)*clslos1_full/(2.0*np.pi), color=colors[1], marker=',',
             label=r'$m_\nu = {}\, \mathrm{{eV}}$'.format(nu_masses_str[1]))
ax1.semilogy(ls, ls*(ls+1)*clslos2_full/(2.0*np.pi), color=colors[2], marker=',',
             label=r'$m_\nu = {}\, \mathrm{{eV}}$'.format(nu_masses_str[2]))

plots += ax1.semilogy(ls, ls*(ls+1)*clsbh0_full/(2.0*np.pi), color=colors[0], marker=',', linestyle='--')
ax1.semilogy(ls, ls*(ls+1)*clsbh1_full/(2.0*np.pi), color=colors[1], marker=',', linestyle='--')
ax1.semilogy(ls, ls*(ls+1)*clsbh2_full/(2.0*np.pi), color=colors[2], marker=',', linestyle='--')
ax1.xaxis.set_visible(False)
ax1.set_ylabel(r'$\ell(\ell+1)C_\ell/2\pi \ \ [\mu K^2]$')
ax1.legend()

ax2.plot(ls, clsbh0_full/clslos0_full, color=colors[0], marker=',', 
         label=r'$m_\nu = {}\, \mathrm{{eV}}$'.format(nu_masses_str[0]))
ax2.plot(ls, clsbh1_full/clslos1_full, color=colors[1], marker=',',
         label=r'$m_\nu = {}\, \mathrm{{eV}}$'.format(nu_masses_str[1]))
ax2.plot(ls, clsbh2_full/clslos2_full, color=colors[2], marker=',',
         label=r'$m_\nu = {}\, \mathrm{{eV}}$'.format(nu_masses_str[2]))
ax2.set_xlabel(r'$\ell$')
ax2.set_ylabel(r'$C_\ell^{BH}/C_\ell^{LoS}$')


ax3.semilogy(ls, ls*(ls+1)*clslos0_full2/(2.0*np.pi), color=colors[0], marker=',', 
                     label=r'$m_\nu = {}\, \mathrm{{eV}}$'.format(nu_masses_str[0]))
ax3.semilogy(ls, ls*(ls+1)*clslos1_full2/(2.0*np.pi), color=colors[1], marker=',',
             label=r'$m_\nu = {}\, \mathrm{{eV}}$'.format(nu_masses_str[1]))
ax3.semilogy(ls, ls*(ls+1)*clslos2_full2/(2.0*np.pi), color=colors[2], marker=',',
             label=r'$m_\nu = {}\, \mathrm{{eV}}$'.format(nu_masses_str[2]))
ax3.semilogy(ls, ls*(ls+1)*clsbh0_full2/(2.0*np.pi), color=colors[0], marker=',', linestyle='--')
ax3.semilogy(ls, ls*(ls+1)*clsbh1_full2/(2.0*np.pi), color=colors[1], marker=',', linestyle='--')
ax3.semilogy(ls, ls*(ls+1)*clsbh2_full2/(2.0*np.pi), color=colors[2], marker=',', linestyle='--')
ax3.xaxis.set_visible(False)
ax3.yaxis.set_visible(False)
ax3.legend()

ax4.plot(ls, clsbh0_full2/clslos0_full2, color=colors[0], marker=',', 
         label=r'$m_\nu = {}\, \mathrm{{eV}}$'.format(nu_masses_str[0]))
ax4.plot(ls, clsbh1_full2/clslos1_full2, color=colors[1], marker=',',
         label=r'$m_\nu = {}\, \mathrm{{eV}}$'.format(nu_masses_str[1]))
ax4.plot(ls, clsbh2_full2/clslos2_full2, color=colors[2], marker=',',
         label=r'$m_\nu = {}\, \mathrm{{eV}}$'.format(nu_masses_str[2]))
ax4.set_xlabel(r'$\ell$')
ax4.yaxis.set_visible(False)

plt.subplots_adjust(hspace=0)
plt.subplots_adjust(wspace=0)

leg = Legend(ax1, plots, ['LoS', 'BH'], loc=[0.7,0.5], frameon=False)
ax1.add_artist(leg) 

plt.savefig('cls_bhlos.pdf')


nu_i = 2
a_array = np.logspace(-5, 0, base=10, num=8000)
zs = 1/a_array - 1
# get the distances corresponding to the redshift array 
distance_cutoffs1 = uts.get_distancesToPresent(0, len(a_array)-1, a_array, 3.0*T_nu*k_B, nu_masses[nu_i])

# pick the range that's important by trial and error 
i_start, i_end = 7000, 7900
distance_cutoffs1 = distance_cutoffs1[i_start:i_end]

cls1_diss_l1 = []

# get the cls at each distance cutoff (hence redshift) 
start_time = time.time()
for discut in distance_cutoffs1: 
    cl = uts.get_clq_LoS(nu_i, q_i, pts_full, None, nu_masses, True, k_magnitudes_full, [1], tau_0_full, discut)
    cls1_diss_l1.append(cl[0])
    
cls1_diss_l1 = np.array(cls1_diss_l1)
print("--- %s seconds ---" % (time.time() - start_time)) 

plt.figure(figsize=(7,5))
cls1_diss_l1pdf = (np.array(cls1_diss_l1[1:])-np.array(cls1_diss_l1[:-1]))/(zs[i_start+2:i_end+1]-zs[i_start+1:i_end])

# plot cls vs z 
plt.plot(zs[i_start+1:i_end], cls1_diss_l1pdf/clslos2_full[0], label=r'$\ell = 1, m_\nu={}$ eV'.format(nu_masses_str[nu_i]), color=colors[0], marker=',')
plt.xlabel(r'z')
plt.ylabel(r'$\frac{1}{C_l}\frac{d C_l}{d z}$')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend()

plt.savefig('dcldz.pdf')

# save array 
np.savez('./arrays/dcls_lmax{}_k{}to{}'.format(l_max, k_min, k_max), cls=cls1_diss_l1, ks=k_magnitudes_full, zs=zs[i_start:i_end])
