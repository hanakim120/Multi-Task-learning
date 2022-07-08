import pandas as pd
import numpy
import matplotlib

raw_data = pd.read_excel('medical_data_prepro_ver1.xlsx',sheet_name = 'Sheet1')
len(raw_data)