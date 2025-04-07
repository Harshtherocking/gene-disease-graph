import pandas as pd
import gzip
import os
from collections import defaultdict

# === CONFIG ===
BASE_DIR = os.getcwd()
INPUT_FILE = os.path.join(BASE_DIR, "DG-Miner_miner-disease-gene.tsv.gz")
OUTPUT_DIR = os.path.join(BASE_DIR, "biane_dgminer")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === STEP 1: Load dataset ===
with gzip.open(INPUT_FILE, 'rt') as f:
    df = pd.read_csv(f, sep='\t', comment='#', header=None, names=["disease", "gene"])

# Assign unique IDs
disease_to_id = {d: i for i, d in enumerate(sorted(df['disease'].unique()))}
gene_to_id = {g: i for i, g in enumerate(sorted(df['gene'].unique()))}

# === STEP 2: Generate user_id.tsv and item_id.tsv ===
user_id_path = os.path.join(OUTPUT_DIR, "user_id.tsv")
item_id_path = os.path.join(OUTPUT_DIR, "item_id.tsv")

pd.DataFrame([(d, i) for d, i in disease_to_id.items()]).to_csv(user_id_path, sep='\t', index=False, header=False)
pd.DataFrame([(g, i) for g, i in gene_to_id.items()]).to_csv(item_id_path, sep='\t', index=False, header=False)

# === STEP 3: Generate train.csv (positive links) ===
train_csv = df.copy()
train_csv['user_id'] = train_csv['disease'].map(disease_to_id)
train_csv['item_id'] = train_csv['gene'].map(gene_to_id)
# Save without header to avoid issues in training script
train_csv[['user_id', 'item_id']].to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False, header=False)

# === STEP 4: Generate adjlist_user_id.tsv and adjlist_item_id.tsv ===
# item_id offset must start after user IDs
num_users = len(disease_to_id)
gene_to_adj_id = {g: i + num_users for g, i in gene_to_id.items()}

adj_user_id_path = os.path.join(OUTPUT_DIR, "adjlist_user_id.tsv")
adj_item_id_path = os.path.join(OUTPUT_DIR, "adjlist_item_id.tsv")

pd.DataFrame([(d, i) for d, i in disease_to_id.items()]).to_csv(adj_user_id_path, sep='\t', index=False, header=False)
pd.DataFrame([(g, i) for g, i in gene_to_adj_id.items()]).to_csv(adj_item_id_path, sep='\t', index=False, header=False)

# === STEP 5: Generate adjlist.txt ===
# Create adjacency list with updated IDs
adjlist = defaultdict(set)

# Add edges
for _, row in df.iterrows():
    d_id = disease_to_id[row['disease']]
    g_id = gene_to_adj_id[row['gene']]
    adjlist[d_id].add(g_id)
    adjlist[g_id].add(d_id)  # For symmetry

# Save adjlist.txt
with open(os.path.join(OUTPUT_DIR, "adjlist.txt"), 'w') as f:
    for node_id in sorted(adjlist.keys()):
        neighbors = " ".join(map(str, sorted(adjlist[node_id])))
        f.write(f"{node_id} {neighbors}\n")
