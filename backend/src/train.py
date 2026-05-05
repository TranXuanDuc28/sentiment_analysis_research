import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from src.utils import save_model, print_banner

def compute_class_weights(labels, num_classes=2):
    valid_labels = [l for l in labels if l >= 0]
    if not valid_labels: return None
    
    from collections import Counter
    counts = Counter(valid_labels)
    total = len(valid_labels)
    
    weights = []
    for i in range(num_classes):
        if counts[i] > 0:
            # sklearn's 'balanced' heuristic: n_samples / (n_classes * np.bincount(y))
            w = total / (len(counts) * counts[i])
        else:
            w = 0.0
        weights.append(w)
        
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

def train_dann(model, tokenizer, source_loader, target_loader, val_loader=None, num_epochs=3, lr=2e-5, device="cpu", class_weights=None):
    from src.utils import EarlyStopping
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    s_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    d_criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=3)
    
    total_steps = num_epochs * len(source_loader)
    current_step = 0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        # DANN requires alternating or joint training. Here we use joint with dynamic lambda.
        target_iter = iter(target_loader)
        
        epoch_s_loss = 0
        epoch_d_loss = 0
        
        pbar = tqdm(source_loader, desc=f"DANN Epoch {epoch}")
        for s_batch in pbar:
            # 1. Prepare Lambda (p increases from 0 to 1), scale max to 0.1 for Transformers
            p = float(current_step) / total_steps
            lambd = (2. / (1. + np.exp(-10 * p)) - 1) * 0.1
            
            # 2. Source batch
            s_input_ids = s_batch["input_ids"].to(device)
            s_attention_mask = s_batch["attention_mask"].to(device)
            s_labels = s_batch["labels"].to(device)
            s_domain_labels = torch.zeros(s_input_ids.size(0), 1, dtype=torch.float).to(device) # Source = 0
            
            # 3. Target batch (unlabeled)
            try:
                t_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                t_batch = next(target_iter)
            
            t_input_ids = t_batch["input_ids"].to(device)
            t_attention_mask = t_batch["attention_mask"].to(device)
            t_domain_labels = torch.ones(t_input_ids.size(0), 1, dtype=torch.float).to(device) # Target = 1
            
            optimizer.zero_grad()
            
            # Forward Source
            s_class_out, s_domain_out = model(s_input_ids, s_attention_mask, alpha=lambd)
            loss_s_class = s_criterion(s_class_out, s_labels)
            loss_s_domain = d_criterion(s_domain_out, s_domain_labels)
            
            # Forward Target
            t_class_out, t_domain_out = model(t_input_ids, t_attention_mask, alpha=lambd)
            loss_t_domain = d_criterion(t_domain_out, t_domain_labels)
            
            # 🚀 TUYỆT CHIÊU: Target Entropy Minimization (Chống sụp đổ cấu trúc cảm xúc)
            # Ép Target phải tự chia làm 2 cụm (Khen/Chê) rõ ràng, không được gom thành 1 cục
            t_probs = torch.softmax(t_class_out, dim=1)
            loss_t_entropy = -torch.mean(torch.sum(t_probs * torch.log(t_probs + 1e-8), dim=1))
            
            # Total Loss (Kết hợp DANN và Entropy)
            total_loss = loss_s_class + (loss_s_domain + loss_t_domain) + 0.1 * loss_t_entropy
            total_loss.backward()
            
            # Gradient Clipping to stabilize Adversarial Training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_s_loss += loss_s_class.item()
            epoch_d_loss += (loss_s_domain.item() + loss_t_domain.item())
            current_step += 1
            pbar.set_postfix(d_loss=f"{epoch_d_loss:.4f}", s_loss=f"{epoch_s_loss:.4f}", alpha=f"{lambd:.2f}")

        if val_loader:
            model.eval()
            val_loss = 0
            correct, total = 0, 0
            with torch.no_grad():
                for b in val_loader:
                    out, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device))
                    val_loss += s_criterion(out, b["labels"].to(device)).item()
                    preds = torch.argmax(out, dim=1)
                    correct += (preds == b["labels"].to(device)).sum().item()
                    total += b["labels"].size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total
            print(f" > DANN Epoch {epoch} - Source Val Loss: {avg_val_loss:.4f} - Source Val Acc: {val_acc:.2f}%")
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                break
    return model
