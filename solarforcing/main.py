import solarforcing.calc as calc
import solarforcing.data_access as data_access
import solarforcing.data_cleaning as data_cleaning
import pydantic
import numpy as np
from datetime import datetime
import xarray as xr

@pydantic.dataclasses.dataclass
class FluxCalculation:
    nbins_e: int = 128
    min_e: float = 30.
    max_e: float = 1000.
    min_lshell: float = 2.
    max_lshell: float = 10.5
    lshell_spacing: float = 0.5
    data_dir: str = 'data/'
    ap_file_url: str = 'ftp://ftp.gfz-potsdam.de/pub/home/obs/Kp_ap_Ap_SN_F107/Kp_ap_Ap_SN_F107_since_1932.txt'
    start_date: datetime = None
    end_date: datetime = None
    angle: float = 80.
    standard_atmosphere_path: str = '../data/msis2.0_atm_out.txt'
    
    def __post_init_post_parse__(self):
        self.energy_grid = calc.gen_energy_grid(self.nbins_e, self.min_e, self.max_e)
        self.lshell = np.arange(self.min_lshell, self.max_lshell, self.lshell_spacing)
        self.glat = calc.lshell_to_glat(self.lshell)

    def grab_data(self):
        self.df = data_access.grab_potsdam_file(self.data_dir, self.ap_file_url, self.start_date, self.end_date)
        self.ap = self.df.ap.values
        self.time = self.df.index
        return self.df
    
    def calculate_flux(self):
        try:
            self.vdk_flux = calc.calculate_flux(self.lshell, self.ap, self.energy_grid, angle=self.angle)
        except (NameError, AttributeError):
            self.grab_data()
            self.vdk_flux = calc.calculate_flux(self.lshell, self.ap, self.energy_grid, angle=self.angle)
        return self.vdk_flux

    def calculate_ipr(self):
        standard_atmosphere = data_access.read_atm(self.standard_atmosphere_path)
        iprm_ds = calc.calculate_iprm(self.vdk_flux, self.glat, self.ap, self.time,
                                      standard_atmosphere.alt.values, standard_atmosphere.rho.values, 
                                      standard_atmosphere.H.values, self.energy_grid)
        
        self.iprm_ds = iprm_ds

        return iprm_ds
    
    def generate_dataset(self):
        
        try:
            isinstance(self.vdk_flux, np.ndarray)
            
        except (NameError, AttributeError):
            self.calculate_flux()
        
        ds = xr.Dataset(
            data_vars=dict(
                vdk_energy_spectrum=(["lshell", "time", "e"], self.vdk_flux),
                lshell=self.lshell,
                glat = self.glat,
                time=self.time,
                e=self.energy_grid,
            ),
            attrs=dict(description="Flux calculation",
                       units="electrons / (cm2 s keV"))

        try:
            isinstance(self.iprm_ds, xr.Dataset)
            
        except (NameError, AttributeError):
            self.calculate_ipr()
        
        self.ds = xr.merge([ds, self.iprm_ds])

        return self.ds
    
@pydantic.dataclasses.dataclass
class SolarIrradiance:
    data_dir: str = 'data/'
    data_url: str = 'https://lasp.colorado.edu/lisird/resources/lasp/nrl2/v02r01/ssi_v02r01_daily_s18820101_e20201231_c20210218.nc'
    start_date: datetime = None
    end_date: datetime = None
    
    def __post_init_post_parse__(self):
        self.raw_ds = data_access.grab_ssi_lasp_file(self.data_dir, self.data_url, self.start_date, self.end_date)
    
    def generate_dataset(self):
        
        # Rename the variables
        self.ds = data_cleaning.rename_variables(self.raw_ds)
        
        # Fix the times and add more date attributes
        self.ds = data_cleaning.center_time(self.ds)
        
        # Add datesecond and date fields
        self.ds = data_cleaning.add_datesec(self.ds)
        self.ds = data_cleaning.add_date(self.ds)
        
        # Add global attributes
        self.ds = data_cleaning.add_global_attrs(self.ds)
        
        # Scale SSI values
        self.ds = data_cleaning.scale_ssi(self.ds)
        
        return self.ds