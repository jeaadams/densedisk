# DenseDisk

DenseDisk is a Python library for analyzing and modeling the physical properties of protoplanetary disks. The library provides tools to compute temperature profiles, sound speeds, surface densities, and azimuthal velocities, as well as to extract velocity components from observational data.

## Features

- Define and calculate detailed temperature profiles of protoplanetary disks.
- Compute surface densities, sound speed profiles, and density distributions.
- Calculate azimuthal velocity components and residuals.
- Extract and interpolate velocity components from observational surface data.

## Installation

To install DenseDisk, clone this repository and install the required dependencies:

```bash
$ git clone https://github.com/jeaadams/densedisk.git
$ cd densedisk
$ pip install -r requirements.txt
```

## Usage

Below is an example demonstrating how to use the `TemperatureProfile` class to extract velocity profiles from disk surface data:

```python
from densedisk.temperature import TemperatureProfile
from astropy import units
import numpy as np

# Define the temperature profile
profile = TemperatureProfile(
    r_profile=(np.linspace(0, 600, 2000) * units.AU).cgs,
    z_profile=(np.linspace(0, 100, 5000) * units.AU).cgs,
    q=0.5,
    r0=(10 * units.AU).cgs,
    T0=40 * units.K,
    M_star=(1.0 * units.M_sun).cgs,
    Sigma_0=189 * units.g / units.cm**2
)

# Extract velocity components
extracted_velocity = profile.extract_velocity_from_surface(
    vphi_real=np.load("path_to_vphi_real.npz"),
    delta_vphi_real=np.load("path_to_delta_vphi_real.npz"),
    surf_12co=np.load("path_to_surf_12co.npz")
)

# Access the extracted velocity data
print(extracted_velocity.vphi_real_r)
print(extracted_velocity.delta_vphi_real_r)
print(extracted_velocity.vphi_real_v)
print(extracted_velocity.delta_vphi_real_v)
print(extracted_velocity.vphi_co)
print(extracted_velocity.delta_vphi_co)
```

## Key Classes

### `Grids`
Represents 2D grids of radial and vertical coordinates in the disk.

- **`r_grid`**: Radial grid (in cm).
- **`z_grid`**: Vertical grid (in cm).

### `Velocity`
Encapsulates azimuthal velocity components.

- **`vphi`**: Azimuthal velocity (in cm/s).
- **`delta_vphi`**: Residual azimuthal velocity (in cm/s).

### `ExtractedVelocity`
Stores the velocity components extracted from surface data.

- **`vphi_real_r`**: Radial coordinates for extracted velocity (in cm).
- **`vphi_real_v`**: Vertical azimuthal velocity components (in cm/s).
- **`vphi_co`**: CO-based azimuthal velocity (in cm/s).
- **`delta_vphi_real_r`**: Radial coordinates for residual velocity (in cm).
- **`delta_vphi_real_v`**: Residual vertical azimuthal velocity components (in cm/s).
- **`delta_vphi_co`**: Residual CO-based azimuthal velocity (in cm/s).

### `TemperatureProfile`
The primary class for modeling the disk's temperature profile and related properties.

#### Attributes:

- **`r_profile`**: Radial coordinates of the disk (in cm).
- **`z_profile`**: Vertical coordinates of the disk (in cm).
- **`q`**: Exponent of the temperature gradient with radius.
- **`r0`**: Reference radius (in cm).
- **`T0`**: Reference temperature at `r0` (in Kelvin).
- **`M_star`**: Mass of the central star (in grams).
- **`Sigma_0`**: Surface density at the reference radius (in g/cm^2).
- **`rho`**: Optional density profile (in g/cm^3).

#### Methods:

- **`calculate_grids()`**: Generate 2D grids for radial and vertical coordinates.
- **`calculate_sigma()`**: Compute the surface density profile.
- **`calculate_T_profile()`**: Calculate the 2D temperature profile.
- **`calculate_cs_profile()`**: Compute the sound speed profile.
- **`calculate_density()`**: Compute the 2D density distribution.
- **`calculate_velocity()`**: Calculate azimuthal velocity components and residuals.
- **`extract_velocity_from_surface()`**: Extract velocity components from surface data.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for enhancements or bug fixes.

## License

DenseDisk is licensed under the [MIT License](LICENSE).

## Acknowledgements

Special thanks to the open-source astronomy and computational physics communities for their invaluable tools and resources.