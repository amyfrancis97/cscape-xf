#%%
# The below code tests if any validation drivers are in the training dataset
import pandas as pd


known_neutrals = pd.read_csv("/Volumes/Samsung_T5/data/neutral_final_all_features.bed", sep = "\t")
known_drivers = pd.read_csv("/Volumes/Samsung_T5/data/gold_standard_with_features.txt", sep = "\t")

df2 = pd.read_csv("/Volumes/Samsung_T5/data/sample_cosmic_gnomad43000.txt", sep = "\t")
df2 = df2.drop(df2.columns[df2.isna().any()].tolist(), axis = 1)

df2["variant_id"] = df2["chrom"] + ":" + df2["pos"].astype(str) + "_" + df2["ref_allele"] + "/" + df2["alt_allele"]
known_drivers["variant_id"] = known_drivers["chrom"] + ":" + known_drivers["pos"].astype(str) + "_" + known_drivers["ref_allele"] + "/" + known_drivers["alt_allele"]

df2 = df2[["variant_id", "grouping"]]
test = known_drivers.merge(df2, on = "variant_id", how = "left")
#%%
df2[["variant_id"]]
#%%
# returns whether or not the variants exist in the training data
res = [test["variant_id"][i] in df2["variant_id"] for i in range(0, len(test))]

if True in res:
    print("Validation data is present in training data")
else:
    print("Validation data is not present in training data")
# %%
known_neutrals = pd.read_csv("/Volumes/Samsung_T5/data/neutral_final_all_features.bed", sep = "\t")
known_drivers = pd.read_csv("/Volumes/Samsung_T5/data/gold_standard_with_features.txt", sep = "\t")

columns_to_keep = list(set(known_neutrals.columns) & set(known_drivers.columns))
# %%
gold_standards_all = pd.concat([known_neutrals[columns_to_keep], known_drivers[columns_to_keep]])
# %%
gold_standards_all.to_csv("/Volumes/Samsung_T5/data/positives_neutrals_gold_standard.txt", sep = "\t", index = None)
# %%
known_drivers
# %%
