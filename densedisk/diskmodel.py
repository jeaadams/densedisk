#!/usr/bin/env python
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pdb

# Constants in cgs units
G = 6.67430e-8         # cm^3 g^-1 s^-2
kb = 1.380649e-16      # erg/K
h = 6.62607015e-27     # erg s
c = 2.99792458e10      # cm/s
mu = 2.37
mh = 1.6726219e-24     # g
AU = 1.495978707e13    # cm
B_e = 57.63596828e9    # Hz

def cs_profile(T_profile):
    """
    Calculate the sound speed (cm/s) from the temperature (K).
    """
    return np.sqrt(kb * T_profile / (mu * mh))

def dcdz(T_profile, z_profile):
    """
    Calculate the derivative of the sound speed with respect to z.
    """
    cs = cs_profile(T_profile)
    return np.gradient(cs, z_profile, axis=0)

def compute_sigma_norm(M_disk, r_profile, R_C=100*AU, p=1, gamma=1, r_gap_center = [50 * AU], gap_width = [5 * AU], gap_depth = [0.9], R_0 = 100 * AU):
    """
    Compute the surface density normalization.
    
    Args:
      M_disk: Disk mass in g.
      r_profile: 1D array of radii in cm.
      R_C: Characteristic radius (cm).
      p, gamma: power-law parameters.
      
    Returns:
      Sigma_c: normalization constant (g/cm^2)
      Sigma_r: surface density profile (g/cm^2)
    """
    # Original power-law profile
    Sigma_profile = (r_profile / R_0)**(-p) * np.exp(- (r_profile / R_C)**gamma)
    
    # Add a Gaussian gap
    if gap_depth is not None:

        for i, gd in enumerate(gap_depth):

            # At center of gap, it drops to 1-gap_depth, far from gap it smooths out
            gap_factor = 1 - gd * np.exp(-0.5 * ((r_profile - r_gap_center[i]) / gap_width[i])**2)

            # Apply the gap to the original surface density profile
            Sigma_profile *= gap_factor

    # Compute normalization so that the integrated surface density equals M_disk
    integrand = 2 * np.pi * r_profile * Sigma_profile
    integral = np.trapz(integrand, r_profile)
    Sigma_c = M_disk / integral
    
    return Sigma_c, Sigma_profile * Sigma_c


def calculate_density(T_profile, r_T, z_T, sigma, M_star=1.989e33):
    """
    Calculate the 2D density profile, rho(r,z) [g/cm^3].
    
    Args:
      T_profile: 2D temperature array (shape: [len(z_profile), len(r_profile)]) in K.
      z_profile: 1D vertical grid (cm).
      r_profile: 1D radial grid (cm).
      M_disk: Disk mass in g.
      M_star: Stellar mass in g.
      
    Returns:
      2D density array rho.
    """

    # Change temperature profile to be one the same grid
    # surf_array = np.column_stack((z_profile, r_profile))
    # interpolator = RegularGridInterpolator((z_T, r_T), T_profile, bounds_error=False)
    # T_grid = interpolator(surf_array)

    # Get sound speed plus gradient
    cs = cs_profile(T_profile)
    dcsdz = dcdz(T_profile, z_T)
    
    # Create grid; shape: (len(z_profile), len(r_profile))
    r_grid, z_grid = np.meshgrid(r_T, z_T)
    gz_star = G * M_star * z_grid / ((r_grid**2 + z_grid**2)**1.5)
    gz_disk = 0  # (disk self-gravity is neglected here)
    gz = gz_star + gz_disk
    
    # Get a rho0
    lnrho_integrand = - (gz / cs**2) - (2 / cs) * dcsdz
    lnrho = cumtrapz(lnrho_integrand, z_T, initial=0, axis=0)
    rho0 = np.exp(lnrho)
    
    integrated_rho = np.trapz(rho0, z_T, axis=0)
    normalization_factor = sigma / integrated_rho
    rho = 0.5 * rho0 * normalization_factor  # factor 0.5 accounts for integration from z=0 to infinity
    return rho

def calculate_vphi(T_profile, r_T, z_T, rho, M_star = 1.989e33):
    """
    Calculate the vphi component (comprised of vstar and epsilon_g)

    Args:
        z_profile(1D array): Vertical height in cm
        r_profile(1D array): Radial distance from the star in cm
        T_profile(2D array): Temperature profile as a function of (r,z) in K
        rho(2D array): Density profile as a function of (r,z) in g/cm^3
        M_star(float): Mass of the star in g. Default is 1 solar mass
        M_disk(float): Mass of the disk in g. Default is 0.01 solar mass

    Returns:
        vphi(2D array): Velocity as a function of (r,z)
        delta_vphi(2D array): Residual velocity as a function of (r,z)
    """
    
    # Create 2D grids for r_profile and z_profile
    r_grid, z_grid = np.meshgrid(r_T, z_T)

    # Calculate v_star^2
    v_star_squared = (G * M_star * r_grid**2 ) / (r_grid**2 + z_grid**2)**(3/2)

    # Calculate epsilon_g
    cs = cs_profile(T_profile)
    Pgas = rho * cs**2
    epsilon_p = (r_T * np.gradient(Pgas, r_T, axis=1, edge_order=2) / rho)

    # Calculate vphi
    vphi = np.sqrt(v_star_squared + epsilon_p)
    delta_vphi = vphi - np.sqrt(v_star_squared)

    return vphi, delta_vphi



def calculate_brightness_temperature(r_profile, z_profile, q=0.5, r0=100*AU, T_m0=12, T_a0=47, az=0.1, wz=0.2):
    """
    Calculate the 2D brightness temperature profile (K).
    
    Args:
      r_profile: 1D radial grid (cm).
      z_profile: 1D vertical grid (cm).
      q: Temperature power-law index.
      r0: Reference radius (cm).
      T_m0: Midplane reference temperature (K).
      T_a0: Atmosphere reference temperature (K).
      az: Midpoint of vertical thermal transition.
      wz: Width of thermal transition.
      
    Returns:
      2D array of brightness temperature (K).
    """
    r_grid, z_grid = np.meshgrid(r_profile, z_profile)
    T_mid = T_m0 * (r_grid / r0)**(-q)
    T_atm = T_a0 * (r_grid / r0)**(-q)
    tanh_arg = ((z_grid / r_grid) - az) / (wz * az)
    ft = 0.5 * np.tanh(tanh_arg) + 0.5
    T = T_mid + ft * (T_atm - T_mid)
    return T




def extract_bt_from_surface(r_profile, z_profile, r_T, z_T, tau_em):
    """
    Extract a brightness temperature profile given a surface 

    Args:
        r (1D Array): radius values from surface profile in cm
        z (1D Array): height values from surface profile in cm

    Returns:
        T (1D Array): model temperature values in K

    """
    # Create model temperature profile (2D Grid)
    model_temps = calculate_brightness_temperature(r_profile=r_T, z_profile=z_T)
    

    # Interpolate given r and z values for the surface
    surf_array = np.column_stack((z_profile, r_profile))
    interpolator = RegularGridInterpolator((z_T, r_T), model_temps, bounds_error=False, fill_value=None)
    t_co = interpolator(surf_array)

    if tau_em is not None:
        T_b = t_co * (1 - np.exp(-tau_em))
    else:
        T_b = t_co

    return T_b
  
 

def extract_vphi_from_surface(r_profile, z_profile, r_T, z_T, T_profile, M_star, rho):
    """
    Extract a velocity and delta velocity profile given a surface 

    Args:
        r_profile (1D Array): model radius values in cm
        z_profile (1D Array): model height values in cm
        r_T (1D Array): radius values for temperature profile in cm
        z_T (1D Array): height values for temperature profile in cm
        T_profile (2D Array): 2D Temperature profile in Kelvin, T(r,z). 

    Returns:
        vphi (1D Array): model velocity values in m/s
        dvphi 1D Array): model delta velocity values in m/s

    """
    # Make model profiles (2D)
    vphi, delta_vphi = calculate_vphi(T_profile = T_profile, r_T = r_T, z_T = z_T, rho = rho, M_star = M_star)

    # Make interpolater to get the 1D profile from the 2D grid
    surf_array = np.column_stack((z_profile, r_profile))
    interpolator = RegularGridInterpolator((z_T, r_T), vphi, bounds_error=False)
    vphi_co = interpolator(surf_array)

    interpolator_deltavphi = RegularGridInterpolator((z_T, r_T), delta_vphi, bounds_error=False)
    delta_vphi_co = interpolator_deltavphi(surf_array)

    return vphi_co, delta_vphi_co

def compute_emission_height(r_profile, r_T, z_T, T_profile, sigma, col_den_file = "Ntau1_T_12CO_J2-1.txt",
                                    nz=2000, x_CO=1e-4, N_dissoc=1e16,
                                   M_star=2 * 2e33,  inclination = 0):
    """
    Compute the tau ~ 0.67 (emission) surface height for a given temperature profile based on Giovanni's code.
    
    Args:
        r (1D array): Radial grid in cm (e.g. np.geomspace(...)*au).
        T_profile: Interpolator for T(z, r). Should be set up on a grid
            with coordinates (z, r) so that calling interp_func(np.column_stack((z, r_val))) returns
            the temperature at vertical positions z for a given radius r_val.
        sigma (1D array): Surface density profile as a function of radius.
        T_crit (1D array): Temperatures for the critical column density interpolation.
        crit_N (1D array): Critical column densities corresponding to T_crit.
        nz (int): Number of vertical points for integration (default: 2000).
        x_CO (float): CO abundance (default: 1e-4).
        N_dissoc (float): Photodissociation column threshold (default: 1e16).
        au, msun, G, mu, m_H: Physical constants.
        
    Returns:
        z_em (1D array): Emission surface height (cm) for each radius.
    """

    interp_func = RegularGridInterpolator((z_T, r_T), T_profile, method = 'nearest',  bounds_error=False, fill_value=None)

    # Load the critical column density vs. temperature (Giovanni's version for now)
    T_crit, crit_N = np.loadtxt(col_den_file, unpack=True)

    sigma_interp = interp1d(r_T, sigma, bounds_error=False, fill_value="extrapolate")
    sigma_on_r_profile = sigma_interp(r_profile)
    
    # Get midplane temperature at z=0 for each r (build a (N,2) array of points: [z, r])
    pts_mid = np.column_stack((np.zeros_like(r_profile), r_profile))
    temp_mid = interp_func(pts_mid)
    cs_mid = cs_profile(temp_mid)
    Omega = np.sqrt(M_star * G / r_profile**3)
    H_mid = cs_mid / Omega 
    rho0 = sigma_on_r_profile / (H_mid * np.sqrt(2 * np.pi))
    z_em = np.zeros_like(r_profile)

    
    for i, radius in enumerate(r_profile):
        # Define a vertical grid from the midplane to the current radius.
        z = np.linspace(0, radius, nz)
        pts_z = np.column_stack((z, np.full_like(z, radius)))
        temp_z = interp_func(pts_z)
        Omega_z = M_star * G / (radius**2 + z**2)**1.5
        cs_z = cs_profile(temp_z)
        exp_int = cumtrapz(Omega_z * z / cs_z**2, z, initial=0)
        rho_z = rho0[i] * (cs_mid[i] / cs_z)**2 * np.exp(-exp_int)
        
        # Compute the CO number density and integrate the column density from the top downward.
        n_CO = x_CO * rho_z / (mu * mh)
        N_col = -cumtrapz(n_CO[::-1], z[::-1], initial=0)
        idx_dissoc = np.searchsorted(N_col, N_dissoc)
        if idx_dissoc >= len(z):
            z_em[i] = 0
            continue
        
        # Interpolate the local critical column density based on temperature.
        N_crit_z = np.interp(temp_z, T_crit, crit_N, left=np.nan, right=np.nan)
        tau_integrand = x_CO * rho_z / (mu * mh) / N_crit_z

        
        try:
            # Integrate tau from the top (largest z) downward and correct for inclination.
            tau = -cumtrapz(tau_integrand[::-1][idx_dissoc:], z[::-1][idx_dissoc:], initial=0)
            tau /= np.cos(np.deg2rad(inclination))
            idx_tau = np.searchsorted(tau, 0.67) + idx_dissoc
            if idx_tau < len(z[::-1]):
                z_em[i] = z[::-1][idx_tau]
    
            else:
                z_em[i] = 0

        except ValueError:
            z_em[i] = 0


    return z_em, 0.67



def generate_disk_profiles(r_profile, r_T, z_T, T_profile, M_disk = 0.01 * 1.989e33,  M_star=1.989e33, gap_depth =None, r_gap_center=None, gap_width=None,  R_C=100*AU, p=1, gamma=1, x_CO = 1e-4, col_den_file = "Ntau1_T_12CO_J2-1.txt", beam_size = 14, N_dissoc = 1e16):
    """
    Generate the key disk profiles in one go.
    
    This function returns:
      - emission_surface: The height (cm) of the emission surface as a function of radius.
      - brightness_temperature: The brightness temperature (K) along the emission surface.
      - residual_velocity: The residual velocity (cm/s) along the emission surface.
    
    Inputs:
      r_profile: 1D radial grid (cm).
      z_profile: 1D vertical grid (cm).
      T_profile: 2D temperature array (K) [shape: len(z_profile) x len(r_profile)].
      M_disk: Disk mass in g.
      M_star: Stellar mass in g.
      beam_size: Beam size of data in AU
    """


    # Calculate surface density profile
    _, sigma = compute_sigma_norm(M_disk, r_profile=r_T, gap_depth = gap_depth, r_gap_center=r_gap_center, gap_width=gap_width, R_C = R_C, p = p, gamma = gamma)

    # Compute the emission surface from the input T_profile
    emission_surface, tau_em = compute_emission_height(r_profile = r_profile, r_T = r_T, z_T = z_T, T_profile = T_profile, sigma = sigma, M_star = M_star, x_CO=x_CO, col_den_file = col_den_file, N_dissoc = N_dissoc)
    
    # Calculate density profile so we only have to do it once
    rho = calculate_density(T_profile = T_profile, sigma = sigma, M_star = M_star, r_T = r_T, z_T = z_T)

    # Convolve the emission surface with a gaussian to account for beam size
    if beam_size is not None:
        sig_beam = beam_size/2.355
        emission_surface_convolved = gaussian_filter1d(emission_surface, sig_beam)

    # # Compute the brightness temperature profile along emission surface
    bt_profile = extract_bt_from_surface(r_profile = r_profile, z_profile = emission_surface, r_T = r_T, z_T = z_T, tau_em = tau_em)
    
    # Compute interpolated residual velocity along the emission surface
    velocity, residual_velocity = extract_vphi_from_surface(r_profile, emission_surface, r_T = r_T, z_T = z_T, T_profile = T_profile, M_star = M_star, rho = rho)
    
    # Correct velocity: Correction factor from Sean Andrews
    r_arcsec = np.array([0.115, 0.145, 0.175, 0.205, 0.235, 0.265, 0.295, 0.325, 0.355, 0.385, 0.415, 0.445, 0.475, 0.505])
    vcorr = np.array([1.1533470253717466, 1.0895965706487443, 1.0592447394943985, 1.03842831610967, 1.0261695008703786, 1.016959348733983, 1.0130694332710568, 1.0097960486074835, 1.00785704434984, 1.0055468373089168, 1.004036629722109, 1.0021529732193727, 1.001156801680209, 1.0010569879459092])

    # Make function to apply
    v_corr_func = interp1d((r_arcsec * 101), vcorr, kind='linear', bounds_error=False, fill_value='extrapolate')
    def v_corr_func_mod(r):
        r = np.asarray(r)
        return np.where(r > 50.5, 1.0, v_corr_func(r))
    
    velocity_corrected = (velocity / 100) * v_corr_func_mod(r_profile / AU) # in m/s
    



    return {
        "emission_surface": emission_surface,
        "emission_surface_convolved": emission_surface_convolved,
        "brightness_temperature": bt_profile,
        "residual_velocity": residual_velocity, # Ignore this for now, want to calculate residuals at the same time as the data
        "velocity": velocity,
        # "tau": tau
    }




