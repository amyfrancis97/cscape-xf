import pandas as pd
import os
import glob

files = glob.glob('/user/home/uw20204/CanDrivR_data/*sample*')
print(files)

df = pd.read_csv(files[0], sep = "\t")

for file in files[1:]:
    df2 = pd.read_csv(file, sep = "\t")
    df = pd.merge(df, df2, how = "outer")

df.to_csv("/user/home/uw20204/CanDrivR_data/sample_merged.bed", sep = "\t", index = None)
