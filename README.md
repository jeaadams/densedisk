# DenseDisk

DenseDisk is a Python library designed to aid in the analysis and modeling of protoplanetary disks. The library currently includes a module for temperature modeling, providing tools to extract velocity profiles from protoplanetary disk surface data using the `TemperatureProfile` class.

## Features

- Define and calculate detailed temperature profiles of protoplanetary disks.
- Compute 1D and 2D profiles for temperature, sound speed, density, and surface density.
- Extract and interpolate velocity profiles from disk surface data.
- Built-in unit support using `astropy` for seamless calculations.

## Installation

To install DenseDisk, clone this repository and install the dependencies listed in `requirements.txt`:

```bash
$ git clone https://github.com/jeaadams/densedisk.git
$ cd densedisk
$ pip install -r requirements.txt
```

## Usage

Below is an example demonstrating how to use the `TemperatureProfile` class to extract velocity profiles from disk surface data.

```python
from densedisk.temperature import TemperatureProfile
from astropy import units
import numpy as np

# Define a temperature profile
temperature_profile = TemperatureProfile(
    r_profile=(np.linspace(0, 600, 2000) * units.AU).cgs,
    z_profile=(np.linspace(0, 100, 5000) * units.AU).cgs,
    q=0.5,
    r0=(10 * units.AU).cgs,
    T0=40 * units.K,
    M_star=(1.0 * units.M_sun).cgs,
    Sigma_0=189 * units.g / units.cm**2,
)

# Extract velocity from disk surface data
extracted_velocity = temperature_profile.extract_velocity_from_surface(
    vphi_real=np.load('path_to_velocity_curve_file.npz'),
    delta_vphi_real=np.load('path_to_residual_velocity_profile.npz'),
    surf_12co=np.load('path_to_surface_data.npz')
)

# Access the extracted velocity components
print(extracted_velocity.vphi_real_r)
print(extracted_velocity.delta_vphi_real_r)
print(extracted_velocity.vphi_real_v)
print(extracted_velocity.delta_vphi_real_v)
print(extracted_velocity.vphi_co)
print(extracted_velocity.delta_vphi_co)
```

## Classes and Methods

### `TemperatureProfile`
This is the primary class for defining and calculating disk profiles. Key parameters include:

- **`r_profile`**: Radial profile of the disk (in cm).
- **`z_profile`**: Vertical profile of the disk (in cm).
- **`q`**: Temperature gradient parameter.
- **`r0`**: Reference radius (in cm).
- **`T0`**: Reference temperature (in Kelvin).
- **`M_star`**: Mass of the central star (in grams).
- **`Sigma_0`**: Surface density at the reference radius (in g/cm^2).

#### Key Methods:

- **`calculate_grids`**: Generates 2D grids for radial and vertical profiles.
- **`calculate_T_profile`**: Computes the 2D temperature profile based on the input parameters.
- **`calculate_cs_profile`**: Calculates the sound speed profile.
- **`calculate_density`**: Computes the 2D density profile by integrating the sound speed and gravitational acceleration.
- **`calculate_velocity`**: Calculates azimuthal velocity (`vphi`) and residuals.
- **`extract_velocity_from_surface`**: Extracts interpolated velocity components from provided surface data.

### `ExtractedVelocity`
The output of `extract_velocity_from_surface`, containing:

- **`vphi_real_r`**: Radial velocity component.
- **`vphi_real_v`**: Vertical velocity component.
- **`vphi_co`**: Velocity from CO line emission.
- **`delta_vphi_real_r`**: Residuals for radial velocity.
- **`delta_vphi_real_v`**: Residuals for vertical velocity.
- **`delta_vphi_co`**: Residuals for CO line velocity.

## Contributing

Contributions to DenseDisk are welcome! If you encounter any issues or have suggestions for new features, feel free to submit an issue or create a pull request.

## License

DenseDisk is released under the [MIT License](LICENSE).

