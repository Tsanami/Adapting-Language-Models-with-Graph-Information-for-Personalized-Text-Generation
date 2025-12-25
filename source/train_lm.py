import json
import pickle
import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer
from tqdm import tqdm

# Импортируем классы из созданного модуля
from project_modules import BaselineT5, GraphAugmentedT5, HybridGraphT5, LMDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["baseline", "grapht5", "hybrid"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Для Mac M1/M2 можно раскомментировать:
    # device = torch("mps") if torch.backends.mps.is_available() else 'cpu'
    
    print("Loading data...")
    # Убедитесь, что пути к файлам верные
    with open("data/dataset.json", "r") as f: 
        data = json.load(f)
    
    emb_dict = None
    if args.model != "baseline":
        with open("data/entity_embeddings.pkl", "rb") as f: 
            emb_dict = pickle.load(f)
        
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    # Разделение данных
    train_data, val_data = train_test_split(data, test_size=0.15, random_state=42)
    
    # Инициализация датасетов
    train_set = LMDataset(train_data, tokenizer, emb_dict, args.model)
    val_set = LMDataset(val_data, tokenizer, emb_dict, args.model)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    
    print(f"Initializing {args.model} model...")
    if args.model == "baseline":
        model = BaselineT5()
    elif args.model == "grapht5":
        model = GraphAugmentedT5()
    else:
        model = HybridGraphT5()
        
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    
    best_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        # Training loop
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1}"):
            optimizer.zero_grad()
            
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbl = batch['labels'].to(device)
            
            if args.model != "baseline":
                g_vec = batch['graph_vector'].to(device)
                outputs = model(ids, mask, g_vec, labels=lbl)
            else:
                outputs = model(ids, mask, labels=lbl)
                
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                lbl = batch['labels'].to(device)
                
                if args.model != "baseline":
                    g_vec = batch['graph_vector'].to(device)
                    outputs = model(ids, mask, g_vec, labels=lbl)
                else:
                    outputs = model(ids, mask, labels=lbl)
                val_loss += outputs.loss.item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f"checkpoints/best_{args.model}.pth")
            print(">>> Saved Best Model")

if __name__ == "__main__":
    main()