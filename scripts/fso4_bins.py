import numpy as np
import pandas as pd
import os

rh_df = pd.read_csv(os.getcwd()+'/data/SO4aerosol-functions-cams2020/secrates_rh_india.csv')
so2_df = pd.read_csv(os.getcwd()+'/data/SO4aerosol-functions-cams2020/secrates_so2_india.csv')
fso4_df = pd.read_csv(os.getcwd()+'/data/SO4aerosol-functions-cams2020/secrates_fso4_india.csv')

# GMT Conversion (+6)
time_mapping = {
    '00': '06',
    '03': '09',
    '06': '12',
    '09': '15',
    '12': '18',
    '15': '21',
    '18': '00',
    '21': '03'
}

rh_df_melted = pd.melt(rh_df, id_vars=['ix', 'iy', 'alon', 'alat'], var_name='month', value_name='rh')
so2_df_melted = pd.melt(so2_df, id_vars=['ix', 'iy', 'alon', 'alat'], var_name='month', value_name='so2')
fso4_df_melted = pd.melt(fso4_df, id_vars=['ix', 'iy', 'alon', 'alat'], var_name='month', value_name='fso4')

merged_df = pd.merge(rh_df_melted,so2_df_melted,on=['ix','iy','alon','alat','month']).merge(fso4_df_melted, on=['ix','iy','alon','alat','month'])
merged_df['hr_GMT'] = merged_df['month'].str[3:5]
merged_df['hr'] = merged_df['hr_GMT'].replace(time_mapping)
merged_df['month'] = merged_df['month'].str[:3]

merged_df['time_category'] = merged_df['hr'].apply(lambda x: 'daytime' if x in ['06', '09', '12', '15'] else 'nighttime')
merged_df['loc_category'] = merged_df['alat'].apply(lambda x: 'north' if x >=23 else 'south')

# Compute bin edges dynamically
num_bins = 5
merged_df['rh_bin'] = pd.cut(merged_df['rh'], 5, labels=[f'bin{i+1}' for i in range(num_bins)])
merged_df['so2_bin'] = pd.cut(merged_df['so2'], 5, labels=[f'bin{i+1}' for i in range(num_bins)])

merged_df = merged_df[['ix', 'iy', 'alon', 'alat','loc_category',
                       'month', 'hr_GMT','hr','time_category',
                         'rh','rh_bin', 'so2', 'so2_bin','fso4']]


merged_df.to_csv(os.getcwd()+'/data/so2_fso4_2020.csv', index=False)