from datetime import datetime

import cf_xarray
import numpy as np
import pandas as pd

def add_global_attrs(ds):
    """
    Add global attributes to SSI dataset, modified from https://svn.code.sf.net/p/codescripts/code/trunk/ncl/solar
    """
    
    current_date = datetime.utcnow().strftime('%Y%m%d')
    ds.attrs['data_script'] = "Created by program createSolarFileNRLSSI2.ncl"
    ds.attrs['data_script_url'] = "https://svn.code.sf.net/p/codescripts/code/trunk/ncl/solar"
    ds.attrs['cesm_contact'] = "Max Grover, mgrover@ucar.edu"
    ds.attrs['creation_date'] = current_date
    ds.attrs['data_source_url'] = "http://lasp.colorado.edu/lisird/data/nrl2_files"
    ds.attrs['data_description_url'] = "http://lasp.colorado.edu/lisird/data/nrl2_files"
    ds.attrs['data_reference']    = \
       "Coddington, O., J. Lean, P. Pilewskie, M. Snow, and D. Lindholm (2016), "+\
       "A solar irradiance climate data record, Bull. Amer. Meteor. Soc., "+\
       "doi:10.1175/BAMS-D-14-00265.1."
    ds.attrs['data_summary']    = "Daily SSI calculated using NRL2 solar irradiance model. "
    "Includes spectrally integrated (total) TSI value. " 
    "This dataset contains spectral solar irradiance as a "
    "function of time and wavelength created with the "
    "Naval Research Laboratory model for spectral and "
    "total irradiance (version 2). Spectral solar irradiance "
    "is the wavelength-dependent energy input to the top of "
    "the Earth's atmosphere, at a standard distance "
    "of one Astronomical Unit from the Sun. Also included "
    "is the value of total (spectrally integrated) solar irradiance."
    
    return ds

def rename_variables(ds):
    """
    Rename variables within the SSI dataset
    """
    variables_to_rename = {'SSI':'ssi',
                           'TSI':'tsi',
                           'TSI_UNC':'tsi_unc',
                           'Wavelength_Band_Width':'band_width'}
    return ds.rename(variables_to_rename)

def _get_tb_name_and_tb_dim(ds):
    """return the name of the time 'bounds' variable and its second dimension"""
    assert 'bounds' in ds.time.attrs, 'missing "bounds" attr on time'
    tb_name = ds.time.attrs['bounds']        
    assert tb_name in ds, f'missing "{tb_name}"'    
    tb_dim = ds[tb_name].dims[-1]
    return tb_name, tb_dim

def center_time(ds, correct_bounds=True):
    """make time the center of the time bounds"""
    ds = ds.copy()
    attrs = ds.time.attrs
    encoding = ds.time.encoding
    
    try:
        tb_name, tb_dim = _get_tb_name_and_tb_dim(ds)
        ds[tb_name][:, 0] = ds.time
        ds[tb_name][:, 1] = ds.time + np.timedelta64(1, 'D')
        ds['time'] = ds[tb_name].compute().mean(tb_dim).squeeze()
        attrs['note'] = f'time recomputed as {tb_name}.mean({tb_dim})'
        
    except AssertionError:
        print('Using default time values')
    
    ds.time.attrs = attrs
    ds.time.encoding = encoding
    return ds

def add_date(ds):
    """
    Adds a date field to a dataset
    """
    ds['date'] = ('time', pd.to_datetime(ds.time).strftime('%Y%m%d').astype(int))
    return ds
    
def add_datesec(ds):
    """
    Add calculation for seconds between the time and 0000 UTC
    """
    ds['datesec'] = ((ds.time - ds.time_bnds[:, 0])/1e9).astype(int)
    ds['datesec'].attrs['units'] = "seconds after midnight UT"
    return ds

def scale_ssi(ds):
    """
    Scale SSI by converting to mW m-2 nm-1
    """
    ds['ssi'] = ds.ssi * 1000
    ds['ssi'].attrs['units'] = "mW m-2 nm-1"
    ds['ssi'].attrs['long_name'] = "NOAA Climate Data Record of Daily Solar Spectral Irradiance"
    return ds