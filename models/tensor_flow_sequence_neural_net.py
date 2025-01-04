#%%
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
#%%
import os
os.chdir("/Users/uw20204/Documents/scripts/CanDrivR/models")
from selected_features import features

#%%
df = pd.read_csv("/Users/uw20204/Documents/data/gnomad_cosmic/sample_cosmic_gnomad14000.txt", sep = "\t")

df2 = df.drop(df.columns[df.isna().any()].tolist(), axis = 1)
#%%
molecular_features = df2[features]
#%%
df2 = df2[["chrom", "pos", "alt_allele"]]
#%%
df2
# %%
from pyfaidx import Fasta

# Path to FASTA file
fasta_file = '/Users/uw20204/Downloads/GRCh38_latest_genomic.fa'

# Load FASTA file
genome = Fasta(fasta_file)

#%%
sequence_report = pd.read_csv("/Users/uw20204/Downloads/sequence_report.tsv", sep = "\t")
#%%
def replace_letter(original_string, index, new_letter):
    if index >= len(original_string):
        return original_string  # Index out of range

    return original_string[:index] + new_letter + original_string[index + 1:]

#%%
def get_flanking_nucleotide_seq(chromosome, position, flanking_length, alt_allele):
    window = int((flanking_length-1)/2)
     
    # Genomic position
    chromosome = chromosome
    position = position
    chromosome = sequence_report.loc[sequence_report["UCSC style name"] == chromosome, "RefSeq seq accession"].item()

    # Extracting the nucleotide
    wild_type = genome[chromosome][position - window - 1:position + window].seq.upper()

    
    mutant = replace_letter(wild_type, window, alt_allele).upper()  # Replace 'ref' with 'alt'

    return wild_type, mutant


import numpy as np
def one_hot_encode(seq):
    # Define a mapping of nucleotides to their one-hot encoding
    mapping = {'A': [1, 0, 0, 0],
               'C': [0, 1, 0, 0],
               'G': [0, 0, 1, 0],
               'T': [0, 0, 0, 1]}

    # Convert each nucleotide to one-hot encoded form
    one_hot_encoded = np.array([mapping[nucleotide] for nucleotide in seq])

    return one_hot_encoded

#%%

def run_model(flanking_length, molecular_features):
    
    nucleotides = [get_flanking_nucleotide_seq(row['chrom'], row['pos'], flanking_length, row["alt_allele"]) for index, row in df2.iterrows()]

    # list of sequences
    wild_type = [nucleotides[i][0] for i in range(0, len(nucleotides))]
    mutant = [nucleotides[i][1] for i in range(0, len(nucleotides))]

    # Apply one-hot encoding to each sequence
    wild_type_encoded_sequences = [one_hot_encode(seq) for seq in wild_type]
    mutant_encoded_sequences = [one_hot_encode(seq) for seq in mutant]

    labels = df["driver_stat"]

    # Flatten the encoded sequences for a standard feedforward network
    WT_flattened_sequences = np.array(wild_type_encoded_sequences).reshape((len(wild_type_encoded_sequences), flanking_length * 4))
    mutant_flattened_sequences = np.array(mutant_encoded_sequences).reshape((len(mutant_encoded_sequences), flanking_length * 4))
    molecular_features = np.array(molecular_features)
    flattened_sequences_combined = np.concatenate([WT_flattened_sequences, mutant_flattened_sequences], axis =1)


    combined_sequences = np.concatenate([WT_flattened_sequences, mutant_flattened_sequences, molecular_features], axis=1)


    X_train, X_test, y_train, y_test = train_test_split(combined_sequences, labels, test_size=0.2, random_state=42)

    # Create model
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(combined_sequences.shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # output layer

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Use binary_crossentropy for binary classification

    # Train model
    model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))  # Using validation_data for testing during training

    # Evaluate model
    accuracy= model.evaluate(X_test, y_test)[1]
    print(f"Accuracy: {accuracy}")
    

    # Predicting labels on the test set
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype("int32")  # Converting probabilities to binary labels

    # Calculating F1 Score and other metrics
    report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
    print(report)
    return flattened_sequences_combined, wild_type, mutant

#%%
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

flattened_sequences_combined, wild_type, mutant = run_model(21, molecular_features)
#%%
flattened_sequences_combined = pd.DataFrame(flattened_sequences_combined)
flattened_sequences_combined.to_csv("/Users/uw20204/Documents/data/gnomad_cosmic/14000_sample_nucleotide_21bp_encoding.txt", sep = "\t", index = None)
#%%
[run_model(i, molecular_features) for i in [5, 11, 21]]
# %%
#[run_model(i) for i in [5, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]]
# %%

#%%
# repeat for the mutant sequence
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer, logging
import torch

#%%
# Pretrained model
checkpoint = 'LongSafari/hyenadna-medium-160k-seqlen-hf'
max_length = 160_000

#%%
from accelerate import *
# bfloat16 for better speed and reduced memory usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
#%%
len(df["driver_stat"].tolist())
#%%
from datasets import Dataset

sequence = wild_type
tokenized = tokenizer(sequence)["input_ids"]
labels = df["driver_stat"].tolist()


from sklearn.model_selection import train_test_split

train_sequences, test_sequences, train_labels, test_labels = train_test_split(
    tokenized, labels, test_size=0.2, random_state=42
)

train_ds = Dataset.from_dict({"input_ids": train_sequences, "labels": train_labels})
test_ds = Dataset.from_dict({"input_ids": test_sequences, "labels": test_labels})

train_ds.set_format("pt")
test_ds.set_format("pt")


# Create a dataset for training
ds = Dataset.from_dict({"input_ids": tokenized, "labels": labels})
ds.set_format("pt")

# Specify the desired metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}

# Set up training arguments including the compute_metrics function
training_args = TrainingArguments(
    output_dir="tmp",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=2e-5,
    evaluation_strategy="epoch",  # to compute metrics at the end of each epoch
)

# Initialize the Trainer with both train_dataset and eval_dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics, 
)

# Train the model
trainer.train()

# Evaluate the model on the evaluation dataset
eval_result = trainer.evaluate()

# Print the evaluation result
print(eval_result)
#%%


# %%
# Print the evaluation result for accuracy
print(f"Accuracy: {eval_result['eval_accuracy']}")
# %%
