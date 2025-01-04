import pandas as pd 

known_drivers = pd.DataFrame({"chrom": ["chr12","chr12", "chr3", "chr17", ]})

known_drivers = pd.read_csv("/Users/uw20204/Documents/data/known_drivers_GrCh38.txt", sep = "\t")
known_drivers = known_drivers['Gene    Chr.    Position    Ref.    Mut.    Confidence_1    Confidence_2    Confidence_3'].str.split(expand = True)
known_drivers = known_drivers.drop([0, 5, 6, 7], axis = 1)

# Create an empty DataFrame to store the expanded rows
expanded_df = []

# Iterate through each row in the original DataFrame
for index, row in known_drivers.iterrows():
    # Extract alleles and remove curly braces
    alleles = row[4].strip('{}').split(',')
    for allele in alleles:

        row[4] = allele
        print(pd.DataFrame(row))
        expanded_df.append(pd.DataFrame(row))

known_drivers_expanded = pd.concat(expanded_df, axis=1).transpose()
known_drivers_expanded.insert(2, "test", known_drivers_expanded[2].tolist())
known_drivers_expanded.to_csv("/Users/uw20204/Documents/data/known_drivers_GrCh38_feat_ext.txt", header = None, sep = "\t", index = None)
