import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Paths
base_dir = os.path.join(os.getcwd(), "biane_dgminer")
train_path = os.path.join(base_dir, "train.csv")

df = pd.read_csv(train_path, names=['user_id', 'item_id'])


# Generate negative samples
num_users = df['user_id'].nunique()
num_items = df['item_id'].nunique()
positive_pairs = set(zip(df['user_id'], df['item_id']))
negatives = []

while len(negatives) < len(df):
    u = np.random.randint(0, num_users)
    i = np.random.randint(0, num_items)
    if (u, i) not in positive_pairs:
        negatives.append((u, i))

# Create full labeled dataset
df_pos = df.copy()
df_pos['label'] = 1

df_neg = pd.DataFrame(negatives, columns=['user_id', 'item_id'])
df_neg['label'] = 0

full_df = pd.concat([df_pos, df_neg], ignore_index=True)

# Shuffle and split
train, temp = train_test_split(full_df, test_size=0.3, random_state=42)
valid, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save
train.to_csv(os.path.join(base_dir, "train.tsv"), sep='\t', index=False, header=False)
valid.to_csv(os.path.join(base_dir, "valid.tsv"), sep='\t', index=False, header=False)
test.to_csv(os.path.join(base_dir, "test.tsv"), sep='\t', index=False, header=False)
