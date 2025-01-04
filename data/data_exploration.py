#%%
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

header = pd.read_csv("/Users/uw20204/Documents/data/cosmic_phenotypes.txt", sep = "\t", nrows = 1)
#%%
df = pd.read_csv("/Users/uw20204/Documents/data/adrenal_gland_cosmic.txt", sep = "\t", names = header.columns.tolist())

df["LOCATION_W_ALLELE"] = df["CHROMOSOME"].astype(str) + ":" + df["GENOME_START"].astype(int).astype(str) + ":" + df["GENOMIC_WT_ALLELE"].astype(str) + "/" + df["GENOMIC_MUT_ALLELE"].astype(str)
df["LOCATION_WITHOUT_ALLELE"] = df["CHROMOSOME"].astype(str) + ":" + df["GENOME_START"].astype(int).astype(str)

# Group by COSMIC_SAMPLE_ID and aggregate LOCATIONs
grouped_df = df.groupby('COSMIC_SAMPLE_ID')['LOCATION_WITHOUT_ALLELE'].apply(list).reset_index()

dataset = grouped_df["LOCATION_WITHOUT_ALLELE"].tolist()

# Transaction Encoder transforms the data into a format for the Apriori algorithm
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df2 = pd.DataFrame(te_ary, columns=te.columns_)

# Applying Apriori: Find frequent itemsets
frequent_itemsets = apriori(df2, min_support=0.005, use_colnames=True)

print(frequent_itemsets)

frequent_itemsets.sort_values("support", ascending = False)
# %%
frequent_itemsets
#%%
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
# Function to check if a sample contains all mutations in an itemset
def contains_itemset(sample_mutations, itemset):
    return itemset.issubset(sample_mutations)

# Iterate over each itemset and create a corresponding column in df
for index, item in frequent_itemsets.iterrows():
    itemset = item['itemsets']
    column_name = 'Itemset_' + '|'.join(map(str, itemset))  # Create a unique column name for the itemset
    # Check if each sample contains the itemset
    df[column_name] = df['LOCATION_WITHOUT_ALLELE'].apply(lambda x: contains_itemset(set(x), itemset))

# Correctly select the itemset columns for X
X = df[[col for col in df.columns if col.startswith('Itemset_')]]
print(X.head())  

# Encode the histology subtype
label_encoder = LabelEncoder()
df['HISTOLOGY_SUBTYPE_CODE'] = label_encoder.fit_transform(df['PRIMARY_HISTOLOGY'])

# Logistic Regression
y = df['HISTOLOGY_SUBTYPE_CODE']

# Fit the model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Output the coefficients
for col, coef in zip(X.columns, model.coef_[0]):
    print(f"{col}: {coef}")

# %%
