import sys
import numpy as np
import diskmodel as d
from astropy import constants as const
from astropy import units as u
from scipy.interpolate import griddata, interp1d
import emcee
import time
from multiprocessing import Pool
import os


# Correction lists from Sean
r_arcsec = np.array([0.115, 0.145, 0.175, 0.205, 0.235, 0.265, 0.295, 0.325, 0.355, 0.385, 0.415, 0.445, 0.475, 0.505])
vcorr = np.array([1.1533470253717466, 1.0895965706487443, 1.0592447394943985, 1.03842831610967, 1.0261695008703786, 1.016959348733983, 1.0130694332710568, 1.0097960486074835, 1.00785704434984, 1.0055468373089168, 1.004036629722109, 1.0021529732193727, 1.001156801680209, 1.0010569879459092])

# Make interpolation function for correction
v_corr_func = interp1d((r_arcsec * 101), vcorr, kind='linear', bounds_error=False, fill_value='extrapolate')

def v_corr_func_mod(r):
    r = np.asarray(r)
    return np.where(r > 50.5, 1.0, v_corr_func(r))

def residual_velocity_model(r, z, v, M_star):
    """
    Calculate the residual velocity profile given a standard Keplerian model
    
    Args:
        r(array): radii in cgs units
        z(array): surface in cgs units
        v(array): velocity in cgs units
    """
    v_star_squared_model = (const.G.cgs.value * M_star * r**2 ) / (r**2 + z**2)**(3/2)
    v_star_model = np.sqrt(v_star_squared_model)
    resv_model = v  - v_star_model
    return resv_model




def run_disk_model(params,
                   r_profile, z_profile, r_T, z_T, T_profile,
                   r_gap_center):
    """
    Turn a params array into (z_model, rv_model).
    """
    R_C          = params[0]
    p            = params[1]
    gamma        = params[2]
    M_disk_model = params[3]
    M_star_model = params[4]
    gap_depths   = [params[5], params[6], params[7]]
    gap_widths   = [params[8], params[9], params[10]]
    

    # Build the model
    _, sigma = d.compute_sigma_norm(M_disk_model, r_profile=r_T, gap_depth = gap_depths, r_gap_center=r_gap_center, gap_width=gap_widths, R_C = R_C * AU, p = p, gamma = gamma)
    rho = d.calculate_density(T_profile, r_T, z_T, sigma, M_star_model )
    velocity, dv = d.extract_vphi_from_surface(r_profile, z_profile, r_T, z_T, T_profile, M_star_model, rho) # in cm/s

    # Velocity model correction
    v_model_corr = (velocity) * v_corr_func_mod(r_profile / AU)
    v_model = residual_velocity_model(r_profile, z_profile, v_model_corr, M_star_model)
    
    return v_model



AU = 1.5e13  # 1 au = 1.5e13 cm

def log_likelihood(params,
                   r_vel_data, v_vel_data, dv_vel_data, z_data,
                   r_profile, z_profile, r_T, z_T, T_profile,
                   r_gap_center):
    # Prior
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf

    # Run model
    v_model = run_disk_model(params, r_profile, z_profile, r_T, z_T, T_profile, r_gap_center)

    # Interpolate onto r_data grid for comparison
    v_model_regrid = np.interp(r_vel_data, r_profile, v_model)

    if np.any(np.isnan(v_model_regrid)):
        return -np.inf
    
    # Calculate resv for data and correct 
    v_data_corr = (v_vel_data) * v_corr_func_mod(r_vel_data / AU)
    v_data = residual_velocity_model(r_vel_data, z_data, v_data_corr, params[4])
    dv_data_corr = (dv_vel_data) * v_corr_func_mod(r_vel_data / AU)


    # Compute chi-squared
    chi2_surf = np.sum(((v_data - v_model_regrid)/dv_data_corr)**2)
    return -0.5 * chi2_surf # log prob



def log_prior(params):
    (R_C, p, gamma,
     M_disk_try, M_star_try,
     gap_depth1, gap_depth2, gap_depth3,
     gap_width1, gap_width2, gap_width3) = params

    # Bounds
    if not (50*AU < R_C < 350*AU):
        return -np.inf
    if not (0.2 < p < 7):
        return -np.inf
    if not (0.2 < gamma < 7):
        return -np.inf
    if not (0.001*1.989e33 <= M_disk_try < 0.5*1.989e33):
        return -np.inf
    if not (1.8*1.989e33 < M_star_try < 2.1*1.989e33): # SWAP FOR GAUSSIAN PRIOR
        return -np.inf
    for gd in [gap_depth1, gap_depth2, gap_depth3]:
        if not (0.1 <= gd <= 0.95):
            return -np.inf
    for gw in [gap_width1, gap_width2, gap_width3]:
        if gw < 0:
            return -np.inf


    return 0.0



def log_posterior(params,
                   r_vel_data, v_vel_data, dv_vel_data, z_data,
                   r_profile, z_profile, r_T, z_T, T_profile,
                   r_gap_center):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params,
                   r_vel_data, v_vel_data, dv_vel_data, z_data,
                   r_profile, z_profile, r_T, z_T, T_profile,
                   r_gap_center)





