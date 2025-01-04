import pandas as pd
import os
import glob

df = pd.read_csv("/user/home/uw20204/CanDrivR_data/cosmic_gnomad_1.csv", sep = "\t")

for i in range(2, 22):
    df2 = pd.read_csv(f'/user/home/uw20204/CanDrivR_data/cosmic_gnomad_{i}.csv', sep = "\t")
    df = pd.merge(df, df2, how = "outer")

df.to_csv("/user/home/uw20204/CanDrivR_data/all_merged.bed", sep = "\t", index = None)
