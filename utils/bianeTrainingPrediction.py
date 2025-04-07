import os
import subprocess

# === Paths ===
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, 'BiANE', 'model')
DATASET_NAME = 'dgminer'
DATA_DIR = os.path.join(ROOT_DIR, 'biane_dgminer')
METAPATH_FILE = os.path.join(DATA_DIR, f'metapath_{DATASET_NAME}.txt')
EMBEDDING_FILE = os.path.join(DATA_DIR, f'emb_{DATASET_NAME}')

# === Step 8.1: Generate Metapaths ===
print("➡️ Generating metapaths...")
subprocess.run([
    'python', 'gen_metapath.py',
    '--dataset', DATASET_NAME,
    '--path_per_node', '10',
    '--path_length', '81'
], cwd=MODEL_DIR, check=True)

# === Step 8.2: Run Metapath2Vec (must be built beforehand) ===
print("➡️ Running metapath2vec...")
metapath2vec_binary = os.path.join(MODEL_DIR, 'code_metapath2vec', 'metapath2vec')

subprocess.run([
    metapath2vec_binary,
    '-train', METAPATH_FILE,
    '-output', EMBEDDING_FILE,
    '-pp', '1',
    '-size', '128',
    '-window', '3',
    '-negative', '5',
    '-threads', '4'
], cwd=MODEL_DIR, check=True)

# === Step 9: Train the model ===
print("🧠 Training the model...")
subprocess.run([
    'python', 'train.py',
    '--dataset', DATASET_NAME
    # optionally add hyperparams here
], cwd=MODEL_DIR, check=True)

# === Step 10: Link Prediction ===
print("🔍 Running link prediction...")
subprocess.run([
    'python', 'link_prediction.py',
    '--dataset', DATASET_NAME
    # optionally add dimensions here
], cwd=MODEL_DIR, check=True)

print("\n✅ All steps completed successfully.")
