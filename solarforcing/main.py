import solarforcing.calc as calc
import solarforcing.data_access as data_access
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
    start_date: datetime = None
    end_date: datetime = None
    angle: float = 80.
    standard_atmosphere_path: str = '../data/msis2.0_atm_out.txt'
    
    def __post_init_post_parse__(self):
        self.energy_grid = calc.gen_energy_grid(self.nbins_e, self.min_e, self.max_e)
        self.lshell = np.arange(self.min_lshell, self.max_lshell, self.lshell_spacing)
        self.glat = calc.lshell_to_glat(self.lshell)

    def grab_data(self):
        self.df = data_access.grab_potsdam_file(self.data_dir, self.start_date, self.end_date)
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
        standard_atmosphere = data_access(self.standard_atmosphere_path)
        
        iprm_ds = calc.calculate_iprm(self.glat, self.ap, self.time,
                                      standard_atmosphere.alt, standard_atmosphere.rho, 
                                      standard_atmosphere.H, self.energy_grid)
        
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