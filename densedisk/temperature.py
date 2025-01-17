from dataclasses import dataclass
from typing import Optional

from astropy import units
import numpy as np
from astropy import constants
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator

@dataclass
class Grids:
    """Representation of 2D grids for the disk's radial and vertical profiles."""

    r_grid: units.quantity.Quantity
    """2D grid representing the radial coordinates in the disk (in cm)."""

    z_grid: units.quantity.Quantity
    """2D grid representing the vertical coordinates in the disk (in cm)."""

@dataclass
class Velocity:
    """Azimuthal velocity components in the disk."""

    vphi: units.quantity.Quantity
    """2D array of azimuthal velocity (in cm/s)."""

    delta_vphi: units.quantity.Quantity
    """2D array of residual azimuthal velocity (in cm/s)."""

@dataclass
class ExtractedVelocity:
    """Extracted velocity components from disk surface data."""

    vphi_real_r: np.ndarray
    """1D array of radial coordinates for extracted velocity (in cm)."""

    vphi_real_v: np.ndarray
    """1D array of vertical azimuthal velocity components (in cm/s)."""

    vphi_co: np.ndarray
    """Interpolated azimuthal velocity from CO line data (in cm/s)."""

    delta_vphi_real_r: np.ndarray
    """1D array of radial coordinates for residual velocity (in cm)."""

    delta_vphi_real_v: np.ndarray
    """1D array of vertical residual velocity components (in cm/s)."""

    delta_vphi_co: np.ndarray
    """Interpolated residual azimuthal velocity from CO line data (in cm/s)."""

@dataclass
class TemperatureProfile:
    """Model of the temperature profile for a protoplanetary disk."""

    r_profile: units.quantity.Quantity
    """1D array representing the radial coordinates of the disk (in cm)."""

    z_profile: units.quantity.Quantity
    """1D array representing the vertical coordinates of the disk (in cm)."""

    q: float
    """Exponent of the temperature gradient with radius."""

    r0: units.quantity.Quantity
    """Reference radius for the temperature profile (in cm)."""

    T0: units.quantity.Quantity
    """Reference temperature at r0 (in Kelvin)."""

    M_star: units.quantity.Quantity
    """Mass of the central star (in grams)."""

    Sigma_0: units.quantity.Quantity
    """Surface density at the reference radius (in g/cm^2)."""

    rho: Optional[units.quantity.Quantity] = None
    """Optional 2D density profile of the disk (in g/cm^3)."""

    def calculate_grids(self) -> Grids:
        """Generate 2D grids for the radial and vertical coordinates."""
        r_grid, z_grid = np.meshgrid(self.r_profile, self.z_profile)
        return Grids(
            r_grid=r_grid,
            z_grid=z_grid,
        )
    
    def calculate_sigma(self) -> units.quantity.Quantity:
        """Calculate the surface density profile (1D array as a function of radius)."""
        # This will eventually be replaced with a calculation
        # But for now, just return the sigma provided
        return self.Sigma_0
    
    def calculate_T_profile(self) -> units.quantity.Quantity:
        """Calculate the 2D temperature profile based on radial and vertical grids."""
        grids = self.calculate_grids()
        return self.T0 * (grids.r_grid/self.r0)**(-self.q)

    def calculate_cs_profile(self):
        """
        Calculate sound speed profile.

        Returns:
            cs_profile: Sound speed profile as a function of (r,z )
        """
        mu = 2.37
        mh = constants.m_p.cgs
        kb = constants.k_B.cgs
        return np.sqrt(kb * self.calculate_T_profile() / (mu * mh))
    
    def calculate_sigma(self, r0 = (10 * units.AU).cgs, p = 1, r_t = (75 * units.AU).cgs, gamma = 1):
        """
        Create a 1D surface density profile based on the r_profile
        Args:
            r_profile(1D array): Radius profile in cm
            Sigma_0(float): in g/cm2
        Returns:
            sigma_r(1D array): Surface density profile as a function of r

        """

        sigma_r_1 = self.Sigma_0 * (self.r_profile/r0)**(-p)
        sigma_r_2 = np.exp((-self.r_profile/r_t))
        sigma_r = sigma_r_1 * sigma_r_2
        return sigma_r.decompose().cgs

    def calculate_dcdz(self):
        """
        Calculate dcs/dz using np.gradient

        Args:
            T_profile(2D Array): 2D temperature profile as a function of (r,z)
            z_profile(1D Array): Heights as a function of r
        Returns:
            dcdz_profile: Derivative of the sound speed wrt z
        """

        # Sound speed profile
        cs = self.calculate_cs_profile()

        return np.gradient(cs, self.z_profile, axis = 0)

    def calculate_density(self):
        """Compute the 2D density profile of the disk."""
        cs = self.calculate_cs_profile()

        # Calculate how sound speed changes as a function of height (2D)
        dcsdz = self.calculate_dcdz()

        # Get sigma(r)
        Sigma_r = self.calculate_sigma()

        # Create 2D grids for r_profile and z_profile
        r_grid, z_grid = np.meshgrid(self.r_profile, self.z_profile)
        # Find gz
        gz_star = constants.G.cgs * self.M_star * z_grid / (r_grid**2 + z_grid**2)**(3/2)
        gz_disk = 0 # 2 * np.pi * G * sigma(r_grid, Sigma_0)
        gz = gz_star + gz_disk

        # This is the integrand
        lnrho_integrand = - (gz/cs**2) - ((2/cs) * dcsdz)

        # Do the integral
        lnrho = integrate.cumtrapz(lnrho_integrand, self.z_profile, initial = 0, axis = 0)

        # Taking the exponent (not normalized yet)
        rho0 = np.exp(lnrho)

        # Integrate rho to get surface density again and calculate normalization factor
        intergal_rho_surf_density = np.trapz(rho0, self.z_profile, axis = 0)
        normalization_factor = Sigma_r / intergal_rho_surf_density

    

        # Apply normalization
        rho = 0.5 * rho0 * normalization_factor


        return rho
    
    def calculate_velocity(self) -> Velocity:
        """
        Calculate the azimuthal velocity components (vphi and delta_vphi).

        Returns:
            Velocity: Object containing vphi and delta_vphi as 2D arrays.
        """

        if self.rho == None:
            rho = self.calculate_density()

        # Create 2D grids for r_profile and z_profile
        grids = self.calculate_grids()

        # Calculate v_star^2
        v_star_squared = (constants.G.cgs * self.M_star * grids.r_grid**2 ) / (grids.r_grid**2 + grids.z_grid**2)**(3/2)

        # Calculate epsilon_g
        cs = self.calculate_cs_profile()
        Pgas = rho * cs**2
        epsilon_p = (self.r_profile * np.gradient(Pgas, self.r_profile, axis=1) / rho).decompose().cgs

        # Calculate vphi
        vphi = np.sqrt(v_star_squared + epsilon_p)
        delta_vphi = vphi - np.sqrt(v_star_squared)

        return Velocity(vphi=vphi, delta_vphi=delta_vphi)

    def extract_velocity_from_surface(
            self,
            vphi_real: np.lib.npyio.NpzFile,
            delta_vphi_real: np.lib.npyio.NpzFile,
            surf_12co: np.lib.npyio.NpzFile,
        ) -> ExtractedVelocity:
        """Extract interpolated velocity components from surface data."""
        velocity = self.calculate_velocity()
        surface_cm = (surf_12co['rz1']) * (101 * units.AU).cgs # convert surface heights from arcsecs to cm
        radius_cm = (surf_12co['rr1']) * (101 * units.AU).cgs
        surf_array = np.column_stack((surface_cm, radius_cm))
        interpolator = RegularGridInterpolator((self.z_profile, self.r_profile), velocity.vphi)
        vphi_co = interpolator(surf_array)

        interpolator_deltavphi = RegularGridInterpolator((self.z_profile, self.r_profile), velocity.delta_vphi)
        delta_vphi_co = interpolator_deltavphi(surf_array)

        # Read in ideal vphi extracted
        vphi_real_r = vphi_real['r'] * 101 * units.AU
        vphi_real_v = vphi_real['v']
        delta_vphi_real_r = delta_vphi_real['r'] * 101 * units.AU
        delta_vphi_real_v = delta_vphi_real['resv']

        return ExtractedVelocity(
            vphi_real_r=vphi_real_r,
            vphi_real_v=vphi_real_v,
            vphi_co=vphi_co,
            delta_vphi_real_r=delta_vphi_real_r,
            delta_vphi_real_v=delta_vphi_real_v,
            delta_vphi_co=delta_vphi_co
        )
