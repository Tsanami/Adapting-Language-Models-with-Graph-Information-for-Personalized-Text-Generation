import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

# --- DATASET ---

class LMDataset(Dataset):
    def __init__(self, data, tokenizer, emb_dict=None, model_type="baseline"):
        self.data = data
        self.tokenizer = tokenizer
        self.emb_dict = emb_dict
        self.model_type = model_type
        
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Формируем промпт в зависимости от типа модели
        if self.model_type == "baseline":
            input_text = f"generate text about {item['entity']}:"
        else:
            input_text = "generate text from graph: " + item['input_graph']
            
        src = self.tokenizer(input_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        tgt = self.tokenizer(item['target_text'], max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        
        batch = {
            "input_ids": src["input_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "labels": tgt["input_ids"].squeeze()
        }
        
        # Добавляем вектор графа, если это не бейзлайн
        if self.model_type != "baseline":
            # Если эмбеддинга нет (OOV), используем нулевой вектор
            vec = self.emb_dict.get(item['entity'], torch.zeros(64).numpy())
            batch["graph_vector"] = torch.tensor(vec, dtype=torch.float32)
            
        return batch

# --- MODELS ---

class BaselineT5(nn.Module):
    def __init__(self, model_name="t5-small"):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def generate(self, input_ids, attention_mask, **kwargs):
        return self.t5.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

class GraphAugmentedT5(nn.Module):
    def __init__(self, model_name="t5-small", graph_dim=64):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.hidden_dim = self.t5.config.d_model
        self.graph_projector = nn.Linear(graph_dim, self.hidden_dim)
        nn.init.xavier_uniform_(self.graph_projector.weight)

    def _prepare(self, ids, mask, g_vec):
        emb = self.t5.shared(ids)
        g_emb = self.graph_projector(g_vec).unsqueeze(1)
        comb_emb = torch.cat([g_emb, emb], dim=1)
        bs = mask.shape[0]
        ones = torch.ones((bs, 1), device=mask.device, dtype=mask.dtype)
        comb_mask = torch.cat([ones, mask], dim=1)
        return comb_emb, comb_mask

    def forward(self, input_ids, attention_mask, graph_vector, labels=None):
        emb, mask = self._prepare(input_ids, attention_mask, graph_vector)
        return self.t5(inputs_embeds=emb, attention_mask=mask, labels=labels)

    def generate(self, input_ids, attention_mask, graph_vector, **kwargs):
        emb, mask = self._prepare(input_ids, attention_mask, graph_vector)
        return self.t5.generate(inputs_embeds=emb, attention_mask=mask, **kwargs)

class HybridGraphT5(nn.Module):
    def __init__(self, model_name="t5-small", graph_dim=64):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.hidden_dim = self.t5.config.d_model
        self.graph_projector = nn.Linear(graph_dim, self.hidden_dim)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        nn.init.xavier_uniform_(self.graph_projector.weight)

    def _fuse(self, ids, mask, g_vec):
        enc_out = self.t5.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        g_emb = self.fusion_norm(self.graph_projector(g_vec).unsqueeze(1))
        fused = torch.cat([enc_out, g_emb], dim=1)
        bs = mask.shape[0]
        ones = torch.ones((bs, 1), device=mask.device, dtype=mask.dtype)
        fused_mask = torch.cat([mask, ones], dim=1)
        return fused, fused_mask

    def forward(self, input_ids, attention_mask, graph_vector, labels=None):
        fused, mask = self._fuse(input_ids, attention_mask, graph_vector)
        return self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=fused), attention_mask=mask, labels=labels)

    def generate(self, input_ids, attention_mask, graph_vector, **kwargs):
        fused, mask = self._fuse(input_ids, attention_mask, graph_vector)
        return self.t5.generate(encoder_outputs=BaseModelOutput(last_hidden_state=fused), attention_mask=mask, **kwargs)