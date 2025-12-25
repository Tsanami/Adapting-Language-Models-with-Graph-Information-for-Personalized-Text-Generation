import json
import pickle
import numpy as np
import torch
import argparse
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--dim", type=int, default=64)
args = parser.parse_args()

def train():
    print("[1/2] Loading triples...")
    with open("data/kg_triples.json", "r") as f:
        triples_list = json.load(f)
        
    triples_np = np.array(triples_list)
    print(f"Training on {len(triples_np)} triples...")
    
    tf = TriplesFactory.from_labeled_triples(triples_np)
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch("mps") if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"[2/2] Training TransE (Dim={args.dim}, Epochs={args.epochs})...")
    result = pipeline(
        training=tf,
        testing=tf,
        model='TransE',
        model_kwargs={'embedding_dim': args.dim},
        optimizer='Adam',
        optimizer_kwargs={'lr': 0.01},
        training_kwargs={'num_epochs': args.epochs},
        device=device,
        random_seed=42
    )
    
    # Извлечение векторов
    model = result.model
    # Получаем матрицу векторов на CPU
    embeddings = model.entity_representations[0](indices=None).detach().cpu().numpy()
    entity_to_id = tf.entity_to_id
    
    # Создаем словарь {EntityName: Vector}
    emb_dict = {name: embeddings[idx] for name, idx in entity_to_id.items()}
    
    with open("data/entity_embeddings.pkl", "wb") as f:
        pickle.dump(emb_dict, f)
        
    print(f"[Done] Saved embeddings for {len(emb_dict)} entities to data/entity_embeddings.pkl")

if __name__ == "__main__":
    train()