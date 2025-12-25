import argparse
import json
import pickle
import torch
import re
import ssl
import pandas as pd
import nltk
import evaluate
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer
from tqdm import tqdm

from project_modules import BaselineT5, GraphAugmentedT5, HybridGraphT5, LMDataset

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# --- МЕТРИКИ ---
print("Loading metrics libraries...")
try:
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
except Exception as e:
    print(f"Error loading metrics: {e}")
    exit()

# --- ФУНКЦИИ ---

def calculate_metrics_graph(model, dataloader, tokenizer, device):
    """Оценка для моделей с графом (Hybrid, GraphT5)"""
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Graph Model", leave=False):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            graph_vec = batch['graph_vector'].to(device)
            
            gen_tokens = model.generate(
                input_ids, mask, graph_vec, 
                max_length=128,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            
            preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            
            labels = batch['labels'].cpu().numpy()
            labels[labels == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            predictions.extend(preds)
            references.extend(refs)

    return compute_scores(predictions, references), predictions

def evaluate_baseline(model, dataloader, tokenizer, device):
    """Оценка для Baseline (без графа)"""
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Baseline", leave=False):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            gen_tokens = model.generate(
                input_ids=input_ids,       
                attention_mask=mask,      
                max_length=128,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            
            preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            
            labels = batch['labels'].cpu().numpy()
            labels[labels == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            predictions.extend(preds)
            references.extend(refs)
            
    return compute_scores(predictions, references), predictions

def compute_scores(predictions, references):
    """Вспомогательная функция для расчета чисел"""
    b = bleu.compute(predictions=predictions, references=[[r] for r in references])
    r = rouge.compute(predictions=predictions, references=references)
    m = meteor.compute(predictions=predictions, references=references)
    
    return {
        "BLEU": b['bleu'],
        "ROUGE-1": r['rouge1'],
        "ROUGE-L": r['rougeL'],
        "METEOR": m['meteor']
    }

def calc_coverage(inputs, preds):
    """Расчет фактического покрытия (Regex)"""
    total, covered = 0, 0
    regex = r"<T>\s*(.*?)(?=\s*(?:\[SEP\]|<H>|$))"
    
    for inp, pred in zip(inputs, preds):
        objs = [m.strip() for m in re.findall(regex, inp) if len(m.strip()) > 0]
        for o in objs:
            total += 1
            if o.lower() in pred.lower():
                covered += 1
                
    return covered / total if total > 0 else 0

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["baseline", "grapht5", "hybrid"], help="Model architecture to evaluate")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading data...")
    try:
        with open("data/dataset.json", "r") as f: data = json.load(f)
    except FileNotFoundError:
        print("Error: data/dataset.json not found.")
        return

    emb_dict = None
    if args.model != "baseline":
        try:
            with open("data/entity_embeddings.pkl", "rb") as f: emb_dict = pickle.load(f)
        except FileNotFoundError:
            print("Error: embeddings not found. Run train_kge.py first.")
            return

    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
    
    _, val_data = train_test_split(data, test_size=0.15, random_state=42)
    
    val_raw_graphs = [item.get('input_graph', "") for item in val_data]
    
    checkpoint_path = f"checkpoints/best_{args.model}.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Initializing {args.model}...")
    
    if args.model == "baseline":
        model = BaselineT5()
        val_dataset = LMDataset(val_data, tokenizer, emb_dict, model_type="baseline")
    elif args.model == "grapht5":
        model = GraphAugmentedT5(graph_dim=64)
        val_dataset = LMDataset(val_data, tokenizer, emb_dict, model_type="graph")
    else: # hybrid
        model = HybridGraphT5(graph_dim=64)
        val_dataset = LMDataset(val_data, tokenizer, emb_dict, model_type="graph")

    print(f"Loading weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    if args.model == "baseline":
        scores, preds = evaluate_baseline(model, val_loader, tokenizer, device)
    else:
        scores, preds = calculate_metrics_graph(model, val_loader, tokenizer, device)

    min_len = min(len(val_raw_graphs), len(preds))
    cov = calc_coverage(val_raw_graphs[:min_len], preds[:min_len])
    
    scores['Model'] = args.model
    scores['Factual Coverage'] = f"{cov*100:.2f}%"
    
    df = pd.DataFrame([scores])
    cols = ['Model', 'BLEU', 'ROUGE-1', 'ROUGE-L', 'METEOR', 'Factual Coverage']
    df = df[cols]
    
    print("\n" + "="*50)
    print(f"RESULTS FOR {args.model.upper()}")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    main()