"""
This module is used to test reading paramters in .cfg 
as an attempt to move parameters into config file format
"""
import pandas as pd

import configparser
config = configparser.ConfigParser()
config.read('config.cfg')

print(type(config['PATH']['data_path']))
print(type(config['TRAINING'].getboolean('sample')))

data_path = config['PATH']['data_path']
df = pd.read_csv(data_path)
print(df.head())