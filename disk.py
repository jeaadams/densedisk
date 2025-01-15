import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy import integrate
import pdb
import cmasher as cmr
import argparse



# Define constants
G = const.G.cgs
mu = 2.37
mh = const.m_p.cgs
kb = const.k_B.cgs

def cs_profile(T_profile):
    """
    Calculate sound speed profile

    Args:
        T_profile(2D Array): 2D temperature profile as a function of (r,z)

    Returns:
        cs_profile: Sound speed profile as a function of (r,z )
    """
    return np.sqrt(kb * T_profile / (mu * mh))


def dcdz(T_profile, z_profile):
    """
    Calculate dcs/dz using np.gradient

    Args:
        T_profile(2D Array): 2D temperature profile as a function of (r,z)
        z_profile(1D Array): Heights as a function of r
    Returns:
        dcdz_profile: Derivative of the sound speed wrt z
    """

    # Sound speed profile
    cs = cs_profile(T_profile)

    dcdz_profile = np.gradient(cs, z_profile, axis = 0)


    return dcdz_profile

def sigma(r_profile, Sigma_0, r0 = (10 * u.AU).cgs, p = 1, r_t = (75 * u.AU).cgs, gamma = 1):
    """
    Create a 1D surface density profile based on the r_profile
    Args:
        r_profile(1D array): Radius profile in cm
        Sigma_0(float): in g/cm2
    Returns:
        sigma_r(1D array): Surface density profile as a function of r

    """

    sigma_r_1 = Sigma_0 * (r_profile/r0)**(-p)
    sigma_r_2 = np.exp((-r_profile/r_t))
    sigma_r = sigma_r_1 * sigma_r_2
    return sigma_r.decompose().cgs


def calculate_density(T_profile, z_profile, r_profile, M_star = (1.0 * u.M_sun).cgs, Sigma_0 =  189 * u.g / u.cm**2):
    """
    Calculate the density profile rho given a 2D temperature and surface density profile

    Args:
        T_profile(2D array): Temperature profile as a function of (r,z) in K
        Sigma_0 (float): Surface density
        z_profile(1D array): Vertical height in cm
        r_profile(float): Radial distance from the star in cm
        M_star(float): Mass of star in g

    Returns:
        rho(2D array): Density as a function of (r,z).
    """

    # Calculate sound speed
    cs = cs_profile(T_profile)

    # Calculate how sound speed changes as a function of height (2D)
    dcsdz = dcdz(T_profile, z_profile)

    # Get sigma(r)
    Sigma_r = sigma(r_profile, Sigma_0)

   # Create 2D grids for r_profile and z_profile
    r_grid, z_grid = np.meshgrid(r_profile, z_profile)
    # Find gz
    gz_star = G * M_star * z_grid / (r_grid**2 + z_grid**2)**(3/2)
    gz_disk = 0 # 2 * np.pi * G * sigma(r_grid, Sigma_0)
    gz = gz_star + gz_disk

    # This is the integrand
    lnrho_integrand = - (gz/cs**2) - ((2/cs) * dcsdz)

    # Do the integral
    lnrho = integrate.cumtrapz(lnrho_integrand, z_profile, initial = 0, axis = 0)

    # Taking the exponent (not normalized yet)
    rho0 = np.exp(lnrho)

    # Integrate rho to get surface density again and calculate normalization factor
    intergal_rho_surf_density = np.trapz(rho0, z_profile, axis = 0)
    normalization_factor = Sigma_r / intergal_rho_surf_density

  

    # Apply normalization
    rho = 0.5 * rho0 * normalization_factor


    return rho


def calculate_vphi(r_profile, z_profile, T_profile, rho = None, M_star = (1.0 * u.M_sun).cgs, Sigma_0 = 189 * u.g / u.cm**2):
    """
    Calculate the vphi component (comprised of vstar and epsilon_g)

    Args:
        z_profile(1D array): Vertical height in cm
        r_profile(float): Radial distance from the star in cm
        T_profile(2D array): Temperature profile as a function of (r,z) in K
        rho(2D array): Density profile as a function of (r,z) in g/cm^3
        M_star(float): Mass of the star in g

    Returns:
        vphi(2D array): Velocity as a function of (r,z)
        delta_vphi(2D array): Residual velocity as a function of (r,z)
    """

    if rho == None:
        rho = calculate_density(T_profile = T_profile, Sigma_0 = Sigma_0, z_profile = z_profile, r_profile = r_profile, M_star = M_star)

    # Create 2D grids for r_profile and z_profile
    r_grid, z_grid = np.meshgrid(r_profile, z_profile)

    # Calculate v_star^2
    v_star_squared = (G * M_star * r_grid**2 ) / (r_grid**2 + z_grid**2)**(3/2)

    # Calculate epsilon_g
    cs = cs_profile(T_profile)
    Pgas = rho * cs**2
    epsilon_p = (r_profile * np.gradient(Pgas, r_profile, axis=1) / rho).decompose().cgs

    # Calculate vphi
    vphi = np.sqrt(v_star_squared + epsilon_p)
    delta_vphi = vphi - np.sqrt(v_star_squared)

    return vphi, delta_vphi






