import numpy as np 
import time  # temp timer 
from classy import Class
from scipy.optimize import fsolve
from scipy import special # bessel functions

# constants 
h = 0.67
c = 299793.  # km/s 
H0 = 100*h
omega_m = 0.27 + 0.049
omega_rad = 2.47e-05/(h*h)
omega_lambda = 1 - omega_m - omega_rad

T_nu = 2.7255*(4/11)**(1/3)*(3.046/3)**(1/4)  # present cnb temperature 
k_B = 8.617333e-05 # eV/K
a_ndec = 1e-10

massless_cutoff = 2e-5   # for whether to treat neutrinos as massless 
scaling = 1e12  # for scaling cl to the correct unit/magnitude 

As_default = 2.215e-9
pivot_default = 5e-2    # default k_pivot 

pivot = 5e-4 
ns = 0.9619
As = As_default*(pivot/pivot_default)**(ns-1)


def get_distanceToPresent(start_index, present_index, a_array, q, m_nu): 
    '''
    Args: 
    start_index (int): index for start time (scale factor) in a_array 
    present_index (int): index for present time in a_array 
    a_array (np.array of float): array of scale factors 
    q (float): q (eV) of neutrino 
    m_nu (float): neutrino mass (eV) 
    
    Return (float): distance (Mpc) traveled by neutrinos between time marked 
    by start_index and the present 
    
    '''
    
    distance_nu = 0 
    
    for i in range (start_index, present_index): 
        da = a_array[i+1]-a_array[i]
        H = H0*np.sqrt(omega_m/a_array[i]**3 + omega_rad/a_array[i]**4 + omega_lambda) 
        
        epsilon = np.sqrt(q**2 + (a_array[i]*m_nu)**2)
        distance_nu += c*(q/epsilon)/(a_array[i]**2*H)*da  # c is to correct the unit to distance 
        
    return distance_nu 


def get_distancesToPresent(start_index, present_index, a_array, q, m_nu): 
    '''
    Optimized version of get_distanceToPresent and returns distances for 
    all indices as an array. 
    
    Args: 
    start_index (int): index for start time (scale factor) in a_array 
    present_index (int): index for present time in a_array 
    a_array (np.array of float): array of scale factors 
    q (float): q (eV) of neutrino 
    m_nu (float): neutrino mass (eV) 
    
    Return (np.array of len (present_index-start_index-1)): distances (Mpc) 
    traveled by neutrinos at times between start_index time and the present
    
    '''
    
    da_array = a_array[start_index+1:present_index+1]-a_array[start_index:present_index]
    H_array = H0*np.sqrt(omega_m/a_array[start_index:present_index]**3+omega_rad/a_array[start_index:present_index]**4+omega_lambda) 
    epsilon_array = np.sqrt(q**2 + (a_array[start_index:present_index]*m_nu)**2)
    
    # distance in each step of integration 
    distances_step = c*(q/epsilon_array)/(a_array[start_index:present_index]**2*H_array)*da_array  # c is to correct the unit to distance 
    distances_sum = np.cumsum(distances_step[::-1])[::-1]  # cumulative sum w/ smallest on right (ex. [1,2,3]->[6,5,3])
    
    return distances_sum 


# outdated but good to see integrals done without the speedup using np.arrays
def get_deltaIntegrand(k, l_index, q_index, pt, nu_masses, nu_index, tau_0): 
    '''
    Args: 
    k (int): k value 
    l_index (int): l index for Cl 
    q_index (int): q index for neutrinos 
    pt: perturbations output from CLASS for k 
    nu_masses (list): neutrino masses 
    nu_index (int): index for neutrino species 
    tau_0: conformal age output from CLASS 
    
    Return ([float, float, float]): all the terms in Delta (including 
    one constant phi term and phi and phiprime integrals) at l_index for the 
    q at q_index and and the nu_index neutrino species 
    
    '''

    a = pt['a']
    tau = pt['tau [Mpc]']
    phi = pt['phi']
    phi_prime = pt['phi_prime']
    qs = np.zeros(n_qbins)
    for q_i in range(n_qbins): 
        qs[q_i] = pt['q_ncdm[{},{}]'.format(nu_index, q_i)][0]*k_B*T_nu  # q now in units of eV 

    m_nu = nu_masses[nu_index] # neutrino mass 
    
    # get the index of the closest value in tau array to tau_0 or tau_ndec 
    taundec_index = np.argmin(np.abs(a - a_ndec)) # neutrino decoupling (take index 0 if times not early enough)
    tau0_index = np.argmin(np.abs(tau - tau_0)) # should be same as len(tau)-1

    # get distances traveled by neutrinos at each time in a array between decoupling and present 
    if (m_nu < massless_cutoff): distancesToPresent = tau[tau0_index]-tau[taundec_index:tau0_index]
    else: distancesToPresent = get_distancesToPresent(taundec_index, tau0_index, a, qs[q_index], m_nu)

    # first term (not an integral)
    phi_const = 0.5*phi[taundec_index]*special.spherical_jn(l_index, k*distanceToPresent) 

    phi_integral = 0 
    phiprime_integral = 0
    
    # calculate the integrals over time steps  
    for i in range (taundec_index, tau0_index): 
        dx = tau[i+1]-tau[i]   # change of variable: x = tau_0-lambda => lambda = tau_0-x 
        da = a[i+1]-a[i]
        H = H0*np.sqrt(omega_m/a[i]**3 + omega_rad/a[i]**4 + omega_lambda)
        epsilon = np.sqrt((qs[q_index])**2 + (a[i]*nu_masses[nu_index])**2)

        j_kdistance = special.spherical_jn(l_index, k*distanceToPresent)
        
        if (nu_masses[nu_index] < massless_cutoff): distanceToPresent = tau[tau0_index]-tau[i+1]
        else: distanceToPresent -= c*(qs[q_index]/epsilon)/(a[i]**2*H)*da  # subtract earliest distance interval from the total

        if (nu_masses[nu_index] < massless_cutoff): 
            phiprime_term = 2*phi_prime[i]*dx*j_kdistance
            phi_term = 0 
        else: 
            phiprime_term = (2+(a[i]*nu_masses[nu_index]/qs[q_index])**2)*phi_prime[i]*dx*j_kdistance
            phi_term = 2*(a[i]*nu_masses[nu_index]/qs[q_index])**2*a[i]*H/c*phi[i]*dx*j_kdistance
        
        phi_integral += phi_term
        phiprime_integral += phiprime_term
        
    return phi_const, phi_integral, phiprime_integral 




def get_deltaIntegrand_opt(k, l_index, q_index, pt, earliest_tf, nu_masses, nu_index, tau_0, distance_cutoff):  
    '''
    Optimized version of get_deltaIntegrand with the implementation of a
    distance cut. 
    
    Args: 
    k (int): k value 
    l_index (int): l index for Cl 
    q_index (int): q index for neutrinos 
    pt: perturbations output from CLASS for k 
    earliest_tfs (tuple or None): phi and psi arguments as (phi, psi) of the earliest transfer fn; if pass in None, takes the first elements in phi and psi from pts to be the earliest 
    nu_masses (list): neutrino masses 
    nu_index (int): index for neutrino species 
    tau_0: conformal age output from CLASS 
    distance_cutoff (float): cutoff for chi (only intergral up to this distance from present) 
    
    Return ([float, float, float]): all the terms in Delta (including 
    one constant phi term and phi and phiprime integrals) at l_index for the 
    q at q_index and the nu_index neutrino species 
    
    '''

    a = pt['a']
    tau = pt['tau [Mpc]']
    phi = pt['phi']
    phi_prime = pt['phi_prime']
    psi = pt['psi']
    
    qs = np.zeros(n_qbins)
    for q_i in range(n_qbins): 
        qs[q_i] = pt['q_ncdm[{},{}]'.format(nu_index, q_i)][0]*k_B*T_nu  # q now in units of eV 

    m_nu = nu_masses[nu_index]
    
    # get the index of the closest value in tau array to tau_0 or tau_ndec 
    taundec_index = np.argmin(np.abs(a - a_ndec)) # neutrino decoupling (take index 0 if times not early enough)
    tau0_index = np.argmin(np.abs(tau - tau_0)) # should be same as len(tau)-1
    
    # get earliest transfer possible 
    if earliest_tf is None: 
        phi_earliest, psi_earliest = phi[taundec_index], psi[taundec_index] 
    else: 
        phi_earliest, psi_earliest = earliest_tf

    # get distances traveled by neutrinos at each time in a array between decoupling and present 
    if (m_nu < massless_cutoff): distancesToPresent = tau[tau0_index]-tau[taundec_index:tau0_index]
    else: distancesToPresent = get_distancesToPresent(taundec_index, tau0_index, a, qs[q_index], m_nu)  

    start_index = taundec_index 
    
    # get the index of first item<distance_cutoff (assuming list in descending order)
    if (distance_cutoff < np.infty): 
        start_index = np.searchsorted(-distancesToPresent, -distance_cutoff) 
        
    # only keep the part whose distance is within the cutoff 
    tau_array = tau[start_index:tau0_index+1]
    a_array = a[start_index:tau0_index+1]
    a_centers = 0.5*(a_array[:-1] + a_array[1:]) 
    phi_prime_centers = 0.5*(phi_prime[start_index:tau0_index] + phi_prime[start_index+1:tau0_index+1])
    phi_centers = 0.5*(phi[start_index:tau0_index] + phi[start_index+1:tau0_index+1])
    psi_centers = 0.5*(psi[start_index:tau0_index] + psi[start_index+1:tau0_index+1])
    dpsi = psi[1:]-psi[:-1]
    dtau = tau[1:]-tau[:-1]
    # to avoid divide by 0 error if we have repeated tau values
    psi_prime = np.divide(dpsi, dtau, out=np.zeros_like(dpsi), where=(dtau != 0.))  
    #psi_prime[np.where(psi_prime == np.inf)] = 0  
    #psi_prime[np.where(psi_prime == -np.inf)] = 0
    
    # calculate the distance to last scattering in case the input pt don't go early enough 
    a_temp = np.logspace(-10, 0, base=10, num=1000) # a_ndec=1e-10
    chi_dec = get_distancesToPresent(0, len(a_temp)-1, a_temp, qs[q_index], m_nu)[0]
    
    # calculate each step of the integral as an array 
    dx_array = tau_array[1:]-tau_array[:-1]  # dtau array
    H_array = H0*np.sqrt(omega_m/a_centers**3+omega_rad/a_centers**4+omega_lambda)

    jk_distances = special.spherical_jn(l_index, k*distancesToPresent[start_index:tau0_index])
    
    # each term carries a - sign when integrate from 0 to dec 
    if (nu_masses[nu_index] < massless_cutoff): 
        # first term (not an integral)
        phi_const = (psi_earliest-0.5*phi_earliest)*special.spherical_jn(l_index, k*chi_dec) 
        # integral terms
        phiprime_terms = (phi_prime_centers + psi_prime)*dx_array*jk_distances
        phi_terms = 0
    else: 
        phi_const = 0.5*phi_earliest*special.spherical_jn(l_index, k*chi_dec)
        phiprime_terms = (2+(a_centers*m_nu/qs[q_index])**2)*phi_prime_centers*dx_array*jk_distances
        phi_terms = 2*(a_centers*m_nu/qs[q_index])**2*a_centers*H_array/c*phi_centers*dx_array*jk_distances
    
    # do integral by sum 
    phi_integral = np.sum(phi_terms)
    phiprime_integral = np.sum(phiprime_terms)
        
    return phi_const, phi_integral, phiprime_integral 


def get_clqcomponents_LoS(nu_index, q_index, pts, earliest_tfs, nu_masses, is_lnk, k_magnitudes, ls, tau_0, distance_cutoff):     
    '''
    Args: 
    nu_index (int): index for neutrino species 
    q_index (int): q index for neutrinos 
    pts: perturbations output from CLASS at all ks  
    earliest_tfs (tuple or None): phi and psi arguments as (phi, psi) of the earliest transfer fn; if pass in None, takes the first elements in phi and psi from pts to be the earliest 
    nu_masses (list): neutrino masses 
    is_lnk (boolean): whether to do dlnk integral (True if assuming Harrison-Zel'dovich-Peebles spectrum) 
    k_magnitudes (np.array): array of k values 
    ls (list): list of l values for which to calculate Cls
    tau_0: conformal age output from CLASS 
    distance_cutoff (float): 
    
    Calculate Cl using line-of-sight integrals.
    
    Return (np.arrays of len(ls)): cls, and the contributions of (phi_constant)^2, (phi_integral)^2, and 
    (phiprime_integral)^2 to the total cl integral. 
    
    '''
    
    l_min = ls[0]
    l_max = ls[-1]
    
    if earliest_tfs is None: 
        phi_earliest = None
    else: 
        # the value at index 0 is not in the correct k range so get rid of it 
        phi_earliest = earliest_tfs['phi'][1:]
        psi_earliest = earliest_tfs['psi'][1:]

    cls = [] 
    phiconst_contribs = []
    phi_contribs = []
    phiprime_contribs = [] 

    # l used for Bessel equations 
    for l_index in range(l_min, l_max+1): 

        cl = 0 
        phiconst_contrib = 0 
        phi_contrib = 0 
        phiprime_contrib = 0

        # calculate the cl integral 
        for i in range(len(k_magnitudes)-1):   # k_index = 0 -- n_kmodes - 1  
            if phi_earliest is None: 
                phi_const, phi_integral, phiprime_integral = get_deltaIntegrand_opt(k_magnitudes[i], l_index, q_index, pts[i], None, nu_masses, nu_index, tau_0, distance_cutoff)
            else: 
                phi_const, phi_integral, phiprime_integral = get_deltaIntegrand_opt(k_magnitudes[i], l_index, q_index, pts[i], [phi_earliest[i], psi_earliest[i]], nu_masses, nu_index, tau_0, distance_cutoff)
            
            # whether to integrate over lnk or k 
            if (is_lnk): 
                dlnk = np.log(k_magnitudes[i+1])-np.log(k_magnitudes[i])
                
                cl += T_nu**2*(4*np.pi)*As_default*(phi_const+phi_integral+phiprime_integral)**2*dlnk 
                phiconst_contrib += T_nu**2*(4*np.pi)*As_default*(phi_const)**2*dlnk
                phi_contrib += T_nu**2*(4*np.pi)*As_default*(phi_integral)**2*dlnk
                phiprime_contrib += T_nu**2*(4*np.pi)*As_default*(phiprime_integral)**2*dlnk
            else: 
                dk = k_magnitudes[i+1]-k_magnitudes[i]
                
                cl += T_nu**2*(4*np.pi)*As_default*k_magnitudes[i]**(ns-2)*(phi_const+phi_integral+phiprime_integral)**2*dk 
                phiconst_contrib += T_nu**2*(4*np.pi)*As_default*k_magnitudes[i]**(ns-2)*(phi_const)**2*dk
                phi_contrib += T_nu**2*(4*np.pi)*As_default*k_magnitudes[i]**(ns-2)*(phi_integral)**2*dk
                phiprime_contrib += T_nu**2*(4*np.pi)*As_default*k_magnitudes[i]**(ns-2)*(phiprime_integral)**2*dk

        # scaling for the correct units
        cls.append(cl*scaling)
        phiconst_contribs.append(phiconst_contrib*scaling)
        phi_contribs.append(phi_contrib*scaling)
        phiprime_contribs.append(phiprime_contrib*scaling)

    return np.array(cls), np.array(phiconst_contribs), np.array(phi_contribs), np.array(phiprime_contribs)


def get_clq_LoS(nu_index, q_index, pts, earliest_tfs, nu_masses, is_lnk, k_magnitudes, ls, tau_0, distance_cutoff, **kwargs):     
    '''
    Args: 
    nu_index (int): index for neutrino species 
    q_index (int): q index for neutrinos 
    pts: perturbations output from CLASS at all ks  
    earliest_tfs (tuple): phi and psi arguments as (phi, psi) of the earliest transfer fn
    nu_masses (list): neutrino masses 
    is_lnk (boolean): whether to do dlnk integral (True if assuming Harrison-Zel'dovich-Peebles spectrum) 
    k_magnitudes (np.array): array of k values 
    ls (list): list of l values for which to calculate Cls
    tau_0: conformal age output from CLASS 
    distance_cutoff (float): 
    
    Calculate Cl using line-of-sight integrals.
    
    Return (np.array): Cls 
    
    '''
    
    if (not is_lnk): 
        k_pivot = kwargs.get('k_pivot', None)

    l_min = ls[0]
    l_max = ls[-1]
    
    if earliest_tfs is None: 
        phi_earliest = None
    else: 
        # the value at index 0 is not in the correct k range so get rid of it 
        phi_earliest = earliest_tfs['phi'][1:]
        psi_earliest = earliest_tfs['psi'][1:]
    
    cls = [] 

    # l used for Bessel equations 
    for l_index in range(l_min, l_max+1): 
        cl = 0 

        # calculate the cl integral 
        for i in range(len(k_magnitudes)-1):   # k_index = 0 -- n_kmodes - 1  
            if phi_earliest is None: 
                phi_const, phi_integral, phiprime_integral = get_deltaIntegrand_opt(k_magnitudes[i], l_index, q_index, pts[i], 
                                                                                    None, nu_masses, nu_index, tau_0, 
                                                                                    distance_cutoff)
            else: 
                phi_const, phi_integral, phiprime_integral = get_deltaIntegrand_opt(k_magnitudes[i], l_index, q_index, pts[i], 
                                                                                [phi_earliest[i], psi_earliest[i]], nu_masses, 
                                                                                nu_index, tau_0, distance_cutoff)
            
            # whether to integrate over lnk or k 
            if (is_lnk): 
                dlnk = np.log(k_magnitudes[i+1])-np.log(k_magnitudes[i])
                cl += (phi_const+phi_integral+phiprime_integral)**2*dlnk 
            else: 
                dk = k_magnitudes[i+1]-k_magnitudes[i]
                cl += k_magnitudes[i]**(ns-2)*(phi_const+phi_integral+phiprime_integral)**2*dk 
               
        # prefactor in cl definition 
        coeff = T_nu**2*(4*np.pi)*As_default
        cl *= coeff 
        
        # scaling for the correct temperature units 
        cls.append(cl*scaling)

    return np.array(cls)


def get_dcl_dlnk(nu_index, q_index, l_index, pts, earliest_tfs, nu_masses, k_magnitudes, tau_0): 
    '''
    Args: 
    nu_index (int): index for neutrino species 
    q_index (int): q index for neutrinos 
    pts: perturbations output from CLASS at all ks
    earliest_tfs (tuple): phi and psi arguments as (phi, psi) of the earliest transfer fn
    nu_masses (list): neutrino masses 
    k_magnitudes (np.array): array of k values
    tau_0: conformal age output from CLASS 
    
    Return (list): list of dcl/dlnk values of dim len(k_magnitudes)-1 
    from line-of-sight integral method 
    '''
    
    # the value at index 0 is not in the correct k range so get rid of it 
    phi_earliest = earliest_tfs['phi'][1:]
    psi_earliest = earliest_tfs['psi'][1:]
    
    dcls = []
    
    for i in range(len(k_magnitudes)-1):   # k_index = 0 -- n_kmodes - 1  
        phi_const, phi_integral, phiprime_integral = get_deltaIntegrand_opt(k_magnitudes[i], l_index, q_index, pts[i], 
                                                                            [phi_earliest[i], psi_earliest[i]], nu_masses, 
                                                                            nu_index, tau_0, np.inf)
        
        dcl = scaling*T_nu**2*(4*np.pi)*As_default*(phi_const+phi_integral+phiprime_integral)**2
        dcls.append(dcl)
    
    return dcls 


def get_clq_BH(q_index, pts, k_magnitudes, ls, is_lnk, n, **kwargs):
    '''
    Args: 
    q_index (int): q index for neutrinos 
    pts: perturbations output from CLASS at all ks  
    k_magnitudes (np.array): array of k values
    ls (list): list of l values for which to calculate Cls
    is_lnk (boolean): whether to do dlnk integral (True if assuming Harrison-Zel'dovich-Peebles spectrum) 
    n (float): n_s value 
    
    Calculate Cl using Boltzmann hierarchy method 
    
    Return (np.arrays): Cls for the specified q_index for all 3 neutrino species
    in the order of their indices
    
    '''
    
    if (not is_lnk): 
        k_pivot = kwargs.get('k_pivot', None)
    
    l_min = ls[0]
    l_max = ls[-1]
    
    Cl0 = []
    Cl1 = []
    Cl2 = []
    
    for l_index in range(l_min, l_max+1):
        Cl0q = 0.0
        Cl1q = 0.0
        Cl2q = 0.0
        
        for k_index in range(len(k_magnitudes)-1):
            pt = pts[k_index]
            a = pt['a']
            a_index = np.where(a>=1.0)[0][0]  # get index of a=1

            Theta0ql = pt['Theta_n_q_l_ncdm[{},{},{}]'.format(0, q_index, l_index)]
            Theta1ql = pt['Theta_n_q_l_ncdm[{},{},{}]'.format(1, q_index, l_index)]
            Theta2ql = pt['Theta_n_q_l_ncdm[{},{},{}]'.format(2, q_index, l_index)]

            if (is_lnk): # assuming ns=1
                delta_lnk = np.log(k_magnitudes[k_index+1])-np.log(k_magnitudes[k_index])
                Cl0q += T_nu**2*(4*np.pi)*(delta_lnk)*As_default*Theta0ql[a_index]*Theta0ql[a_index]
                Cl1q += T_nu**2*(4*np.pi)*(delta_lnk)*As_default*Theta1ql[a_index]*Theta1ql[a_index]
                Cl2q += T_nu**2*(4*np.pi)*(delta_lnk)*As_default*Theta2ql[a_index]*Theta2ql[a_index]
            else: 
                delta_k = k_magnitudes[k_index+1]-k_magnitudes[k_index]
                Cl0q += T_nu**2*(4*np.pi)*(delta_k/k)*As*(k/k_pivot)**(n-1)*Theta0ql[a_index]*Theta0ql[a_index]
                Cl1q += T_nu**2*(4*np.pi)*(delta_k/k)*As*(k/k_pivot)**(n-1)*Theta1ql[a_index]*Theta1ql[a_index]
                Cl2q += T_nu**2*(4*np.pi)*(delta_k/k)*As*(k/k_pivot)**(n-1)*Theta2ql[a_index]*Theta2ql[a_index]
                
        Cl0.append(Cl0q*scaling)
        Cl1.append(Cl1q*scaling)
        Cl2.append(Cl2q*scaling)

    return np.array(Cl0), np.array(Cl1), np.array(Cl2)


def get_clqindep(nu_index, q_indices, pts, earliest_tfs, nu_masses, k_magnitudes, ls, tau_0): 
    '''
    Args: 
    nu_index (int): index for neutrino species 
    q_indices (int): q indices to integrate over to get q-indep average 
    pts: perturbations output from CLASS at all ks  
    earliest_tfs (tuple): phi and psi arguments as (phi, psi) of the earliest transfer fn 
    nu_masses (list): neutrino masses 
    k_magnitudes (np.array): array of k values 
    ls (list): list of l values for which to calculate Cls
    tau_0: conformal age output from CLASS 
    
    Return (np.array of len(ls)): Cls averaged over the given range of q over q_indices
    
    '''

    l_min = ls[0]
    l_max = ls[-1]
    
    # the value at index 0 is not in the correct k range so get rid of it 
    phi_earliest = earliest_tfs['phi'][1:]
    psi_earliest = earliest_tfs['psi'][1:]
    
    cls = [] 
    
    # get the q/T values used for integration 
    qs = np.arange(1.5, 1.5*(n_qbins+1), 1.5)
    qs = qs[q_indices]
    dq = qs[1:] - qs[:n_qbins-1]

    # l used for Bessel equations 
    for l_index in range(l_min, l_max+1): 
        cl = 0 

        # calculate the cl integral 
        for i in range(len(k_magnitudes)-1):   # k_index = 0 -- n_kmodes - 1
            deltaqs = []
            
            for q in q_indices: 
                phi_const, phi_integral, phiprime_integral = get_deltaIntegrand_opt(k_magnitudes[i], l_index, q, pts[i],
                                                                                    [phi_earliest[i], psi_earliest[i]], 
                                                                                    nu_masses, nu_index, tau_0, np.infty)
                deltaqs.append(phi_const+phi_integral+phiprime_integral)  
            
            # integrate over q 
            delta = 2/3/special.zeta(3)*np.sum(deltaqs[:n_qbins-1]*qs[:n_qbins-1]**2*dq/(np.exp(qs[:n_qbins-1])+1))
            
            # integrate over k 
            dlnk = np.log(k_magnitudes[i+1])-np.log(k_magnitudes[i])
            cl += delta**2*dlnk 
               
        # prefactor in cl definition 
        coeff = T_nu**2*(4*np.pi)*As_default
        cl *= coeff 
        
        # scaling for the correct units 
        cls.append(cl*scaling)

    return np.array(cls)



def run_class(parameters, gettransfer):
    '''
    Run CLASS with the input parameters and return the perturbations and 
    the value of tau_0 (which should be fixed but we still return the value 
    for the purpose of checking) and the earliest transfer (if asked). Print the
    amount of time taken to run. 
    
    Args: 
    parameters: parameters to run CLASS 
    gettransfer (boolean): whether to get the earliest transfer 
    
    Return: (pts, tau_0) if gettransfer=False and  (pts, tau_0, transfer) otherwise
    '''

    start_time = time.time()

    cosmo = Class()
    cosmo.set(parameters)
    cosmo.compute()

    pts = cosmo.get_perturbations()['scalar']
    tau_0 = cosmo.get_current_derived_parameters(['conformal_age'])['conformal_age']
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # 45999 is the largest redshift possible 
    if (gettransfer): 
        tf = cosmo.get_transfer(45999)
        return pts, tau_0, tf 
    
    return pts, tau_0