# Báo cáo Toàn diện (12 Kịch bản: Multidomain & Multilingual)

### Chặng 1: Multidomain (RQ1)
| Mã | Kỹ thuật | F1-Macro |
| :--- | :--- | :--- |
| **S1** | MD Baseline | 0.8815 |
| **S1A** | MD Single-source | 0.8107 |
| **S1B** | MD Few-shot | 0.8914 |
| **S2** | MD Multi-task | 0.8941 |
| **S3** | MD DANN (XLM-R) | 0.8839 |

### Chặng 2: Multilingual (RQ2)
| Mã | Kỹ thuật | F1-Macro |
| :--- | :--- | :--- |
| **S0** | Mono VI Baseline | 0.9628 |
| **S4** | ML Zero-shot (XLM-R) | 0.7853 |
| **S4B** | ML Few-shot | 0.9395 |
| **S5** | ML Translation | 0.7493 |
| **S6** | ML Joint | 0.9583 |

### Chặng 3: Unified Framework (RQ3)
| Mã | Kỹ thuật | F1-Macro |
| :--- | :--- | :--- |
| **S7** | Unified Zero-shot | 0.8319 |
| **S8** | Unified DANN (XLM-R) | 0.9053 |
| **S9** | Unified Multi-task | 0.8610 |

### Chặng 4: Model Ablation (mBERT vs XLM-R)
| Mã | Kỹ thuật | F1-Macro |
| :--- | :--- | :--- |
| **S3** | MD DANN (XLM-R) | 0.8839 |
| **S10** | MD DANN (mBERT) | 0.8351 |
| **S4** | ML Zero-shot (XLM-R) | 0.7853 |
| **S11** | ML Zero-shot (mBERT) | 0.7519 |
| **S8** | Unified DANN (XLM-R) | 0.9053 |

