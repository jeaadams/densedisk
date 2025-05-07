from diskmodel import generate_disk_profiles
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

AU =  1.5e13   # au to cm conversion

# Define radial and vertical grids in cm (avoid zero radius to prevent division issues)
r_profile = np.linspace(1, 600, 2000) * AU  

# Disk and stellar masses in grams
M_disk = 0.3 * 1.989e33  
M_star = 2.1 * 1.989e33  

# Load temperature profile of choice
rs, zs, T_profile = np.load('../ancillary/hd163296_exponential.npy')
r_T = rs[0, :] * AU # Get just 1D r vals
z_T = zs[:, 0] * AU # Get just 1D z vals
T_profile[T_profile > 1000] = 1000 # Set temperatures above 1000 to 1000


# Now specify disk parameters
N_dissoc = 6e16
x_CO_12co =  1e-4 * 20
col_den_file_12co = f"../ancillary/Ntau1_T_12CO_J2-1.txt"
beam_size = (0.1257 * 101) # 0.14 arcsec approximately for robust = 0.5
gap_depth = [0.98, 0.92, 0.98]
r_gap_center = [48 * AU, 86*AU, 145*AU]
gap_width = [8*AU, 10*AU, 40*AU]
R_C = 200 * AU
gamma = 1
p = 0.5


###### GENERATE DISK MODEL ######
profiles_12co =  generate_disk_profiles(r_profile = r_profile, r_T = r_T, z_T = z_T, T_profile = T_profile, M_disk = M_disk, M_star = M_star, x_CO = x_CO_12co,  r_gap_center=r_gap_center, gap_width=gap_width, gap_depth = gap_depth, col_den_file = col_den_file_12co, beam_size = beam_size, R_C = R_C , p = p, gamma = gamma, N_dissoc = N_dissoc)


## Plot
fig = plt.figure(figsize=(6, 7))
gs = gridspec.GridSpec(nrows=3, ncols=1)
plt.subplots_adjust(hspace=0)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

# Top plot: Surface
ax1.plot(r_profile / AU, profiles_12co["emission_surface_convolved"] / AU, color='k', lw=1, label='Surface', ls='-')
ax1.set_ylabel("Surface Height [AU]")
ax1.legend(loc='upper left', frameon=False)
ax1.tick_params(labelbottom=False)

# Middle plot: Brightness temperature
ax2.plot(r_profile / AU, profiles_12co["brightness_temperature"], color='k', lw=1, label='Brightness Temperature', ls='-')
ax2.set_ylabel("Brightness Temerature [K]")
ax2.legend(loc='upper left', frameon=False)
ax2.tick_params(labelbottom=False) 

# Bottom plot: Velocity Profile
ax3.plot(r_profile / AU, profiles_12co["velocity"] / 100, color='k', lw=1, label='Velocity', ls='-')
ax3.set_ylabel("Velocity [m/s]")
ax3.set_xlabel("Radius [AU]")
ax3.legend(loc='upper left', frameon=False)

plt.show()
plt.savefig("profiles.png")

