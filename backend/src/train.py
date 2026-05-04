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

def train_model(model, tokenizer, train_loader, val_loader=None, num_epochs=5, lr=3e-5, device="cuda", class_weights=None):
    """Huấn luyện mô hình với Validation và Early Stopping"""
    from src.utils import EarlyStopping
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    early_stopping = EarlyStopping(patience=2)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in progress:
            optimizer.zero_grad()
            outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation Step
        if val_loader:
            model.eval()
            val_loss = 0
            correct, total = 0, 0
            with torch.no_grad():
                for b in val_loader:
                    out = model(b["input_ids"].to(device), b["attention_mask"].to(device))
                    lgt = out[0] if isinstance(out, tuple) else out
                    loss = criterion(lgt, b["labels"].to(device))
                    val_loss += loss.item()
                    preds = torch.argmax(lgt, dim=1)
                    correct += (preds == b["labels"].to(device)).sum().item()
                    total += b["labels"].size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total
            print(f" > Epoch {epoch} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.2f}%")
            
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                break
    return model

def train_dann(model, tokenizer, source_loader, target_loader, val_loader=None, num_epochs=5, lr=3e-5, device="cuda", class_weights=None):
    """Huấn luyện DANN với Validation trên Source Sentiment"""
    from src.utils import EarlyStopping
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    s_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    d_criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=2)

    total_steps = num_epochs * min(len(source_loader), len(target_loader))
    current_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        progress = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)), desc=f"DANN Epoch {epoch}")
        
        for s_batch, t_batch in progress:
            p = float(current_step) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            current_step += 1
            optimizer.zero_grad()
            
            # Source Forward
            s_logits, sd_logits = model(s_batch["input_ids"].to(device), s_batch["attention_mask"].to(device), alpha=alpha)
            loss_s_sentiment = s_criterion(s_logits, s_batch["labels"].to(device))
            loss_s_domain = d_criterion(sd_logits, s_batch["domain_ids"].to(device))
            
            # Target Forward
            _, td_logits = model(t_batch["input_ids"].to(device), t_batch["attention_mask"].to(device), alpha=alpha)
            loss_t_domain = d_criterion(td_logits, t_batch["domain_ids"].to(device))
            
            total_loss = loss_s_sentiment + loss_s_domain + loss_t_domain
            total_loss.backward()
            optimizer.step()
            progress.set_postfix(s_loss=f"{loss_s_sentiment.item():.4f}", d_loss=f"{(loss_s_domain + loss_t_domain).item():.4f}")
        
        # Validation Step (Source Sentiment)
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for b in val_loader:
                    out, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device))
                    loss = s_criterion(out, b["labels"].to(device))
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f" > DANN Epoch {epoch} - Source Val Loss: {avg_val_loss:.4f}")
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                break
    return model
