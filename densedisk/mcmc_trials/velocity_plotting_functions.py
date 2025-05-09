import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import corner
from astropy import units as u
from mcmc_velocity_profile import run_disk_model

def plot_walkers(chain):
    
    params = ['$R_C$', '$p$', '$\gamma$', r'$M_{disk}$', r'$M_{*}$', r'Gap 1 Depth', r'Gap 2 Depth', r'Gap 3 Depth', r'Gap 1 Width',  r'Gap 2 Width',  r'Gap 3 Width', r'$x_{CO}$']

    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(12, 1, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], hspace=0)

    # Number of steps for the x-axis
    steps = np.arange(chain.shape[0])

    # Plot each parameter in a separate subplot
    for i in range(chain.shape[2]):
        ax = plt.subplot(gs[i, 0])
        for walker in range(chain.shape[1]):
            ax.plot(steps, chain[:, walker, i], lw=0.5)
        ax.set_ylabel(f'{params[i]}')
        ax.set_xlabel('N Steps')

    # Adjusting layout
    plt.tight_layout()

def plot_corner(chain_array):
  

    # Reshape the samples into a (nsamples, nparams) array
    # Flatten the first two dimensions
    samples = chain_array.reshape(-1, chain_array.shape[2])

    # Plot
    figure = corner.corner(samples, labels=['$R_C$', '$p$', '$\gamma$', r'$M_{disk}$', r'$M_{*}$', r'Gap 1 Depth', r'Gap 2 Depth', r'Gap 3 Depth', r'Gap 1 Width',  r'Gap 2 Width',  r'Gap 3 Width', r'$x_{CO}$'],
                       quantiles=[0.16, 0.5, 0.84],  # Shows the quantiles if you want
                       show_titles=True, title_kwargs={"fontsize": 20})


def select_random_model(chain,r_profile, z_profile, r_T, z_T, T_profile, r_gap_center, noexp=False):
    """
    Select a random z model fit to the emission surface 
    
    Args:
        chain (np.array): Array of shape (nsteps, nwalkers, nvalues)
        r values (np.array): r values to fit to. 
    """
    # Select step and walker index
    step_index = np.random.randint(chain.shape[0])
    walker_index = np.random.randint(chain.shape[1])

    # Get the parameters
    selected_parameters = chain[step_index, walker_index]
    
    # Use these parameters in model
    model = run_disk_model(selected_parameters,
                   r_profile, z_profile, r_T, z_T, T_profile,
                   r_gap_center)

    
    return model



def unpack_params(params):
    R_C = params[0]
    p = params[1]
    gamma = params[2]
    M_disk = params[3]
    M_star = params[4]
    gap_depths = [params[5], params[6], params[7]]
    gap_widths = [params[8], params[9], params[10]]
    x_CO = params[11]
    
    return R_C, p, gamma, M_disk, M_star, gap_depths, gap_widths, x_CO