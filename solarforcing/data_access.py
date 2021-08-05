import os

import urllib
import pandas as pd

def grab_potsdam_file(data_dir = 'data/', start_date=None, end_date=None):
    """Grabs solar weather data from potsdam ftp server
    
    Parameters
    ----------
    data_dir: `str`, optional
       Data directory - by default, creates a "data" directory in the current directory
    
    start_date: `datetime`, optional
       Start date, by default, set to None providing all values since 1932
    
    end_date: `datetime`, optional
       End date, by default, set to None providing all values up to the current date
    
    """
    
    
    # Check to see if the data directory exists, if not, create it
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    file, message = urllib.request.urlretrieve('ftp://ftp.gfz-potsdam.de/pub/home/obs/Kp_ap_Ap_SN_F107/Kp_ap_Ap_SN_F107_since_1932.txt', 'data/Kp_ap_Ap_SN_F107_since_1932.txt')
    
    # read in the dataset using pandas
    df = pd.read_csv(file, skiprows=39, header=0, delim_whitespace=True)
    
    # Create a datetime index using the year, month, day columns
    cols=["#YYY","MM","DD"]
    df['date'] = df[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    df.index = pd.to_datetime(df.date)
    
    # Subset the data if needed
    if not (start_date is None):
        df = df[df.index >= start_date]
    
    if not (end_date is None):
        df = df[df.index <= end_date]

    # Use the 3-hourly data to calculate an average daily value, round to three decimal places
    df['ap'] = df[['ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6', 'ap7', 'ap8']].mean(axis=1).round(3)
    
    
    return df[['ap']]
    
    