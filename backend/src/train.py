import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from src.utils import save_model, print_banner

def compute_class_weights(labels):
    valid_labels = [l for l in labels if l >= 0]
    if not valid_labels: return None
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(valid_labels), y=valid_labels)
    return torch.tensor(weights, dtype=torch.float)

def train_model(model, tokenizer, train_loader, num_epochs=3, lr=2e-5, device="cuda", class_weights=None):
    """Huấn luyện mô hình chuẩn (Single-task)"""
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in progress:
            optimizer.zero_grad()
            # Handle both BaseModel and DANNModel
            outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")
    return model

def train_dann(model, tokenizer, source_loader, target_loader, num_epochs=3, lr=2e-5, device="cuda", class_weights=None):
    """Huấn luyện DANN với Dynamic Alpha"""
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    s_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    d_criterion = nn.CrossEntropyLoss()

    total_steps = num_epochs * min(len(source_loader), len(target_loader))
    current_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        progress = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)), desc=f"DANN Epoch {epoch}")
        
        for s_batch, t_batch in progress:
            # Tính Alpha động: tăng dần từ 0 lên 1
            p = float(current_step) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            current_step += 1

            optimizer.zero_grad()
            
            # 1. Source Forward
            s_input_ids = s_batch["input_ids"].to(device)
            s_attn = s_batch["attention_mask"].to(device)
            s_labels = s_batch["labels"].to(device)
            s_domain = s_batch["domain_ids"].to(device)
            
            s_logits, sd_logits = model(s_input_ids, s_attn, alpha=alpha)
            loss_s_sentiment = s_criterion(s_logits, s_labels)
            loss_s_domain = d_criterion(sd_logits, s_domain)
            
            # 2. Target Forward
            t_input_ids = t_batch["input_ids"].to(device)
            t_attn = t_batch["attention_mask"].to(device)
            t_domain = t_batch["domain_ids"].to(device)
            
            _, td_logits = model(t_input_ids, t_attn, alpha=alpha)
            loss_t_domain = d_criterion(td_logits, t_domain)
            
            # Total Loss
            total_loss = loss_s_sentiment + loss_s_domain + loss_t_domain
            
            total_loss.backward()
            optimizer.step()
            progress.set_postfix(s_loss=f"{loss_s_sentiment.item():.4f}", d_loss=f"{(loss_s_domain + loss_t_domain).item():.4f}", alpha=f"{alpha:.4f}")
            
    return model
