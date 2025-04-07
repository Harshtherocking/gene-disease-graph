import numpy as np
import pandas as pd
import pickle
import os

# Set path to your data folder
base_dir = os.path.join(os.getcwd(), "biane_dgminer")

# Load user and item ID files
user_df = pd.read_csv(os.path.join(base_dir, "user_id.tsv"), sep='\t', header=None)
item_df = pd.read_csv(os.path.join(base_dir, "item_id.tsv"), sep='\t', header=None)

num_users = len(user_df)
num_items = len(item_df)

# Use a constant vector (e.g., all 1's or any fixed values) of dimension 128
user_attr = np.ones((num_users, 128), dtype=np.float32)
item_attr = np.ones((num_items, 128), dtype=np.float32)

# Save attributes as .pkl files
with open(os.path.join(base_dir, "user_attr.pkl"), 'wb') as f:
    pickle.dump(user_attr, f)

with open(os.path.join(base_dir, "item_attr.pkl"), 'wb') as f:
    pickle.dump(item_attr, f)

print("Constant attribute files saved successfully.")
