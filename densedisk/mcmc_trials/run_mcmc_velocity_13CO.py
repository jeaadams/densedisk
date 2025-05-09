import sys
sys.path.insert(0, '../')
import numpy as np
import diskmodel as d
from mcmc_velocity_profile import log_posterior
from astropy import constants as const
from astropy import units as u
from scipy.interpolate import griddata, interp1d
import emcee
import time
from multiprocessing import Pool
import os


def main():

    AU = 1.5e13  # 1 au = 1.5e13 cm
    t0 = time.time()
    r_gap_center = [48*AU, 86*AU, 145*AU]
    ndim = 11
    nthreads = 6
    nsteps = 20000
    ninits = 500
    nwalk = 64
    if nthreads > 1: os.environ["OMP_NUM_THREADS"] = "1"

    

    # Load T_profile grid (Anna)
    rs, zs, T_profile = np.load('../ancillary/hd163296_exponential.npy')
    r_T = rs[0, :] * AU
    z_T = zs[:, 0] * AU
    T_profile[T_profile > 1000] = 1000

    # r_profile for the model (a fine radial grid)
    r_profile = np.linspace(20, 350, 2000) * AU
    

    # Data
    surf = np.load('/Users/jea/HD163296/surf_data/HD163296_13CO.robust_0.5_ideal.npz')

    # Put in cgs units
    r_surf = surf['rr1'] * 101 * AU
    z_surf = surf['rz1'] * 101 * AU

    v_data = np.load('/Users/jea/HD163296/velocity_tests/velocity_curve_13CO_iters5_fitmethodSHO.npz')
    r_vel = v_data["r"] * 101 * AU
    v_vel = v_data["v"] * 100 # Convert to cm/s
    dv_vel = v_data["dv"] * 100
    z_profile = np.interp(r_profile, r_surf, z_surf)
    z_data = np.interp(r_vel, r_surf, z_surf)

    guess = [200*AU, 0.8, 1.2, 
                1e-2*1.989e33, 2.0*1.989e33, 
                0.5, 0.5, 0.5,
                15*AU, 25*AU, 35*AU] 

    # Initialize walker positions
    p0 = guess * (1 + 0.01*np.random.randn(nwalk, ndim))  # ~1% scatter around guess

    # Initialization sampling
    with Pool(processes=nthreads) as pool:
        isampler = emcee.EnsembleSampler(nwalk, ndim, log_posterior, pool=pool,
                                        args=(r_vel, v_vel, dv_vel, z_data,
                r_profile, z_profile, r_T, z_T, T_profile,
                r_gap_center))
        isampler.run_mcmc(p0, ninits)

    # Prune stray walkers
    isamples = isampler.get_chain()
    lop0 = np.quantile(isamples[-1,:,:], 0.25, axis=0)
    hip0 = np.quantile(isamples[-1,:,:], 0.75, axis=0)
    p00 = [np.random.uniform(lop0, hip0, ndim) for iw in range(nwalk)]

    # Run MCMC
    with Pool(processes=nthreads) as pool:
        sampler = emcee.EnsembleSampler(nwalk, ndim, log_posterior, pool=pool,
                                        args=(r_vel, v_vel, dv_vel, z_data,
                r_profile, z_profile, r_T, z_T, T_profile,
                r_gap_center))
        sampler.run_mcmc(p00, nsteps, progress=True)

    # flattened chain after burn-in
    chain = sampler.get_chain()
    print(chain.shape)
    
    # get log-posteriors
    lnprob = sampler.get_log_prob()

    # Save the posterior samples
    ofile = f'velocity_fit_13CO.npz'
    np.savez(ofile, chain=chain, lnprob=lnprob)
    t1 = time.time()
  
    # progress monitoring
    print('Velocity in %.1f seconds\n' % (t1 - t0))


if __name__ == '__main__':
    main()


