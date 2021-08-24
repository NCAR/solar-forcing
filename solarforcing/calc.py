import numpy as np
from numba import jit, jit_module
import pydantic
from datetime import datetime
from scipy import integrate

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

@jit(nopython=True)
def fang(y, Emono):
# Input: 
# y - normalized atmospheric column mass as a function of vertical location (z)
# Emono - is incident electron energy (keV)
# Output:
# f - quanity calculated by eqn. (4)

    # Table 1.
    p1 = np.array([(1.24616E+0,  1.45903E+0, -2.42269E-1,  5.95459E-2), 
                   (2.23976E+0, -4.22918E-7,  1.36458E-2,  2.53332E-3),
                   (1.41754E+0,  1.44597E-1,  1.70433E-2,  6.39717E-4),
                   (2.48775E-1, -1.50890E-1,  6.30894E-9,  1.23707E-3),
                   (-4.65119E-1, -1.05081E-1, -8.95701E-2,  1.22450E-2),
                   (3.86019E-1,  1.75430E-3, -7.42960E-4,  4.60881E-4),
                   (-6.45454E-1,  8.49555E-4, -4.28581E-2, -2.99302E-3),
                   (9.48930E-1,  1.97385E-1, -2.50660E-3, -2.06938E-3)])

    # terms in eq. (5)
    lne = np.log(Emono)
    lne2 = lne*lne
    lne3 = lne*lne2

    # step 2. calculate the C array in (5)
    c = np.empty((8))
    for i in range(8):
        c[i] = np.exp(p1[i,0] + p1[i,1]*lne + p1[i,2]*lne2 + p1[i,3]*lne3)
        
    # eq. (4) - Normalized energy deposition
    f = c[0]*y**c[1]*np.exp(-c[2]*y**c[3]) + c[4]*y**c[5]*np.exp(-c[6]*y**c[7])
    
    return f

@jit(nopython=True)
def iprmono(e, flux, rho, H):
    # assign constants
    epsilon = 0.035 # keV 

    ipr = np.empty((e.size,rho.size))

    for i,energy in enumerate(e): # loop over energy index

        # step 1. (eq. 1) 
        y = (2/energy)*(rho*H/6.0e-6)**0.7
        f = np.empty(y.size)

        for j,yy in enumerate(y):
            f[j] = fang(yy, energy)

        # calculate ipr (qtot) using eq. (3) for a specified flux at ea. energy
        Qmono = flux[i]*energy # (keV cm−2 s−1)
        ipr[i,:] = f*Qmono/(epsilon*H)

    return ipr

@jit(nopython=True)
def calculate_ipr(glats, aps, alt, rho, H, e):
    ipr_vals = np.empty(shape=(len(aps),len(glats), len(e), len(alt)))
    for i in range(len(aps)):
        for j in range(len(glats)):
            l = 1.01 / np.cos(glats[j]*np.pi/180.)**2
            # calculate the top of the atmosphere energetic electron energy spectrum
            flux_sd = vdk2016(e, l, aps[i])
            
            # van de Kamp is per steradian (electrons / (cm2 sr s keV))
            # assume flux is isotropic inside a nominal bounce loss cone (BLC) angle
            # of 66.3˚. The area of the BLC in sr is 2pi(1-cosd(66.3))
            flux = 2.*np.pi*(1-np.cos(np.radians(80))) * flux_sd

            # calculate the IPR as a function f height and energy
            ipr = iprmono(e, flux, rho, H)
            
            # Add the results to the ipr array
            ipr_vals[i, j, :, :] = ipr
            
    return ipr_vals

jit_module(nopython=True, error_model="numpy")

def calculate_iprm(glats, aps, times, alt, rho, H, e):
    """
    Calculates iprm from a dataset
    
    Parameters
    ----------
    glats: numpy.array
      Array of geomagnetic latitudes
    aps: numpy.array
      Array of AP values
    times: list
      List of times corresponding to ap values
    alt: numpy.array
      Array of altitude values
    rho: numpy.array
      Array of density values
    H: numpy.array
      Array of scaling height values
    e: numpy.array
      Energy grid used for this calcualtion

   Returns
   -------
   xarray.Dataset with IPRM values

    """
    
    ipr = calculate_ipr(glats, aps, alt=alt, rho=rho, H=H, e=e)
    
    # calculate ipr total by integrating across the energy spectrum
    ipr_tot = integrate.simps(ipr, e, axis=2)
    
    # calculate iprm
    iprm = ipr_tot/rho
    
    # Calculate pressure levels
    plevs = 1013.*np.exp(-alt/6.8)
    
    # Build a dataset for the output
    out_ds = xr.Dataset(data_vars=dict(iprm=(['time', 'glat', 'pressure'], iprm),
                                       ap = (['time'], aps),
                                       glat = glats,
                                       pressure = plevs,
                                       time=times))
    out_ds = out_ds.transpose("time", "pressure", "glat")
    
    return out_ds.sortby('pressure')

