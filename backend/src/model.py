import torch
import torch.nn as nn
from transformers import AutoModel
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class BaseModel(nn.Module):
    """Mô hình phân loại cảm xúc tiêu chuẩn (không có DANN)"""
    def __init__(self, model_name="xlm-roberta-base", num_labels=2):
        super(BaseModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.sentiment_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.sentiment_head(pooled_output)
        return logits, None

class DANNModel(nn.Module):
    """Mô hình Domain Adversarial Neural Network"""
    def __init__(self, model_name="xlm-roberta-base", num_labels=2):
        super(DANNModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Sentiment Head
        self.sentiment_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )
        
        # Domain Head (Binary: Source vs Target)
        self.domain_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, alpha=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Sử dụng [CLS] token (vị trí 0)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Cảm xúc (Sentiment)
        s_logits = self.sentiment_head(pooled_output)
        
        # Miền (Domain) - Nếu có alpha thì thực hiện Reverse Gradient
        if alpha is not None:
            reverse_feature = GradientReversalFunction.apply(pooled_output, alpha)
            d_logits = self.domain_head(reverse_feature)
            return s_logits, d_logits
        else:
            return s_logits, None

class AdvancedMultiTaskModel(nn.Module):
    """Mô hình Multi-task Learning với 3 tác vụ: Sentiment, Domain, và Language"""
    def __init__(self, model_name="xlm-roberta-base", num_labels=2, num_domains=6, num_languages=2):
        super(AdvancedMultiTaskModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Sentiment Head
        self.sentiment_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )
        
        # Domain Head
        self.domain_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_domains)
        )
        
        # Language Head
        self.language_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_languages)
        )

    def forward(self, input_ids, attention_mask, alpha_domain=None, alpha_language=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Sentiment
        s_logits = self.sentiment_head(pooled_output)
        
        # Domain
        if alpha_domain is not None:
            reverse_feature_domain = GradientReversalFunction.apply(pooled_output, alpha_domain)
            d_logits = self.domain_head(reverse_feature_domain)
        else:
            d_logits = self.domain_head(pooled_output)
            
        # Language
        if alpha_language is not None:
            reverse_feature_lang = GradientReversalFunction.apply(pooled_output, alpha_language)
            l_logits = self.language_head(reverse_feature_lang)
        else:
            l_logits = self.language_head(pooled_output)
            
        return s_logits, d_logits, l_logits

