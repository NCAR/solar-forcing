import numpy as np
from numba import jit, jit_module
import pydantic
from datetime import datetime

@jit(nopython=True)
def gen_energy_grid(nbins, min_e=30., max_e=1000.):
    """Generate a grid of energies for the flux spectrum
    
    This function generates an energy grid given a number of grids
    
    Parameters
    ----------
    nbins: `float`
       The number of logarithmically spaced grid points
       
    min_e: `float`, optional
       Minimum energy range, defaults to 30 keV
    
    max_e: `float`, optional
       Maximum energy range, defaults to 1000 keV
       
    Returns
    -------
    `numpy.array`
       Resultant energy grid given the parameters specified

    """

    e1 = np.log10(min_e)
    e2 = np.log10(max_e)
    e = 10**(e1 + (e2-e1)*np.arange(nbins)/(nbins-1))
    
    return e

@jit(nopython=True)
def vdk2016(e, l, Ap):
    """A routine (vdk2016) to return an energy spectrum for a specific l-shell and Ap
    
    Calculations here are defined by van de Kamp et al. 2016 (doi:10.1002/2015JD024212)
    
    Parameters
    ----------
    e: `numpy.array`
       An energy grid, as defined in gen_energy_grid
       
    l: `float`
       Lshell value
       
    Ap: `float`
       Ap value
    
    Returns
    -------
    `numpy.array`
       Flux spectral density with the same shape as e

    """

    lpp = -0.7430*np.log(Ap) + 6.5257
    Spp = l - lpp

    # vdK2016 eqn.(8)

    A = 8.2091*Ap**0.16255
    b = 1.3754*Ap**0.33042
    c = 0.13334*Ap**0.42616
    s = 2.2833*Ap**-0.22990
    d = 2.7563e-4*Ap**2.6116

    # integral flux >30 keV (F30) electrons / (cm2 sr s)
    F30 = np.exp(A) / (np.exp(-b*(Spp-s)) + np.exp(c*(Spp-s)) + d)

    # vdK2016 eqn.(9)

    E = 3.3777*Ap**-1.7038 + 0.15
    bk = 3.7632*Ap**-0.16034
    sk = 12.184*Ap**-0.30111

    k = -1.0 / (E*np.exp(-bk*Spp) + 0.30450*np.cosh(0.20098*(Spp-sk))) - 1
    
    # solve eqn 3 for C
    # C is an offset, and k is the spectral gradient
    x=k+1
    c = F30*x/(1e3**x-30.**x)
    
    # calcualte the spectral density of the flux S(E) = CE^k
    # in electrons / (cm2 sr s keV)
    flux_spectral_density = e**k*c
    
    return flux_spectral_density

@jit(nopython=True)
def calculate_flux(lshell, aps, e, angle=80.):
    """Calculates a solar flux using vdk2016
    
    Uses lshell, ap values, and e to calculate a flux
    
    Parameters
    ----------
    lshell: `numpy.array`
       Lshell values, one dimensional.
    
    aps: `numpy.array`
       Ap values, one dimensional.
    
    e: `numpy.array`
       Energy grid, two dimensional.
       
    angle: `float`, optional
       Angle to use when coverting to per steradian, if not given defaults
       to 80 degrees.
    
    Returns
    -------
    `numpy.array`
       Energy flux, using vdk2016 calculation

    """
    flux = np.empty(shape=(len(lshell),len(aps), len(e)))
    for i in range(len(lshell)):
        for j in range(len(aps)):
            # calculate the top of the atmosphere energetic electron energy spectrum
            
            if aps[j] != 0:
                flux_sd = vdk2016(e, lshell[i], aps[j])
                
            else:
                flux_sd = np.nan * e
            
            # van de Kamp is per steradian (electrons / (cm2 sr s keV))
            # assume flux is isotropic inside a nominal bounce loss cone (BLC) angle
            # of 80˚. The area of the BLC in sr is 2pi(1-cosd(66.3))
            flux[i, j, :] = 2.*np.pi*(1-np.cos(np.radians(angle))) * flux_sd
            
    return flux

# Convert from lshell to glat
@jit(nopython=True)
def lshell_to_glat(lshell):
    """Converts lshell to geomagnetic latitude
    
    Parameters
    ----------
    lshell: `float`, `numpy.array`
       Lshell value
    
    Returns
    -------
    `numpy.array`
       Geomagnetic latitude
    
    """
    return np.arccos((2.02/lshell) - 1) * (90./np.pi)

@jit(nopython=True)
def glat_to_lshell(glat):
    """Converts geomagnetic latitude to lshell
    
    Parameters
    ----------
    glat: `float`, `numpy.array`
       Geomagnetic latitude
    
    Returns
    -------
    `numpy.array`
       Lshell
    
    """
    return 1.01 / np.cos(glat*np.pi/180.)**2

jit_module(nopython=True, error_model="numpy")

