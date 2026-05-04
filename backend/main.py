
import argparse
import yaml
import os
import torch
from transformers import AutoTokenizer
from src.dataset import load_amazon_split, load_vsfc, load_tweeteval, load_multi_domain_amazon, make_dataloader
from src.model import BaseModel, DANNModel
from src.train import train_model, train_dann, compute_class_weights
from src.evaluate import evaluate_model
from src.visualize_embeddings import visualize_tsne
from src.utils import print_banner, save_results, print_dataset_statistics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, default=0, help="Scenario to run (0=all, 1-5)")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. SCENARIO 1: Monolingual Baselines
    if args.s in [0, 1]:
        print_banner("Scenario 1a: Baseline English (XLM-R)")
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        train_texts, train_labels, train_d_ids = load_amazon_split("english", "books", "train", max_samples=config["scenarios"]["max_samples_train"])
        print_dataset_statistics(train_labels, "Amazon Books (EN Train)")
        
        train_loader = make_dataloader(train_texts, train_labels, train_d_ids, tokenizer, batch_size=config["training"]["batch_size"], shuffle=True)
        model = BaseModel(config["model"]["name"])
        weights = compute_class_weights(train_labels)
        
        model = train_model(model, tokenizer, train_loader, num_epochs=int(config["training"]["epochs"]), lr=float(config["training"]["learning_rate"]), device=device, class_weights=weights)
        torch.save(model.state_dict(), "checkpoints/model_en_books.pt")
        
        test_texts, test_labels, test_d_ids = load_amazon_split("english", "books", "test", max_samples=config["scenarios"]["max_samples_test"])
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, tokenizer, batch_size=config["training"]["batch_size"])
        res_s1a = evaluate_model(model, test_loader, device, "S1a_Baseline_EN")
        save_results(res_s1a, "results/results_s1a.json")

        print_banner("Scenario 1b: Baseline Vietnamese (XLM-R vs PhoBERT)")
        # XLM-R on VI
        train_texts_vi, train_labels_vi, train_d_ids_vi = load_vsfc("train", max_samples=config["scenarios"]["max_samples_train"])
        train_loader_vi = make_dataloader(train_texts_vi, train_labels_vi, train_d_ids_vi, tokenizer, batch_size=config["training"]["batch_size"], shuffle=True)
        model_vi = BaseModel(config["model"]["name"])
        weights_vi = compute_class_weights(train_labels_vi)
        model_vi = train_model(model_vi, tokenizer, train_loader_vi, num_epochs=config["training"]["epochs"], device=device, class_weights=weights_vi)
        
        test_texts_vi, test_labels_vi, test_d_ids_vi = load_vsfc("test", max_samples=config["scenarios"]["max_samples_test"])
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, tokenizer, batch_size=config["training"]["batch_size"])
        res_s1b_xlmr = evaluate_model(model_vi, test_loader_vi, device, "S1b_XLMR_VI")
        
        # PhoBERT on VI (Optional but highly recommended)
        try:
            print("\nComparing with PhoBERT baseline...")
            phobert_tok = AutoTokenizer.from_pretrained(config["model"]["vi_baseline"])
            phobert_model = BaseModel(config["model"]["vi_baseline"])
            # PhoBERT needs segmented text, but for a quick baseline we'll use raw
            train_loader_ph = make_dataloader(train_texts_vi, train_labels_vi, train_d_ids_vi, phobert_tok, batch_size=config["training"]["batch_size"], shuffle=True)
            phobert_model = train_model(phobert_model, phobert_tok, train_loader_ph, num_epochs=config["training"]["epochs"], device=device, class_weights=weights_vi)
            test_loader_ph = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, phobert_tok, batch_size=config["training"]["batch_size"])
            res_s1b_pho = evaluate_model(phobert_model, test_loader_ph, device, "S1b_PhoBERT_VI")
        except Exception as e:
            print(f"PhoBERT baseline failed: {e}")

    # 2. SCENARIO 2: Zero-shot Cross-lingual (EN -> VI)
    if args.s in [0, 2]:
        print_banner("Scenario 2: Zero-shot Cross-lingual (English -> Vietnamese)")
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        model = BaseModel(config["model"]["name"])
        model.load_state_dict(torch.load("checkpoints/model_en_books.pt", map_location=device))
        
        test_texts_vi, test_labels_vi, test_d_ids_vi = load_vsfc("test", max_samples=config["scenarios"]["max_samples_test"])
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, tokenizer, batch_size=config["training"]["batch_size"])
        res_s2 = evaluate_model(model, test_loader_vi, device, "S2_ZeroShot_EN_VI")
        save_results(res_s2, "results/results_s2.json")
        
        # Vẽ t-SNE để xem không gian vector Anh-Việt có khớp nhau không
        print("\n[Visualization] Drawing t-SNE for S2...")
        test_en_t, test_en_l, test_en_d = load_amazon_split("english", "books", "test", max_samples=300)
        ld_en = make_dataloader(test_en_t, test_en_l, test_en_d, tokenizer, batch_size=16)
        visualize_tsne(model, tokenizer, [ld_en, test_loader_vi], ["EN Books", "VI VSFC"], device, "S2_CrossLingual_Alignment")

    # 3. SCENARIO 3: Unseen Domain Transfer (Books+Electronics -> Apparel)
    if args.s in [0, 3]:
        print_banner("Scenario 3: Unseen Domain Transfer (Books+Electronics -> Apparel)")
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        # Train on Books and Electronics
        t1, l1, d1 = load_amazon_split("english", "books", "train", max_samples=config["scenarios"]["max_samples_train"]//2)
        t2, l2, d2 = load_amazon_split("english", "electronics", "train", max_samples=config["scenarios"]["max_samples_train"]//2)
        train_texts = t1 + t2
        train_labels = l1 + l2
        train_d_ids = d1 + d2
        
        train_loader = make_dataloader(train_texts, train_labels, train_d_ids, tokenizer, batch_size=config["training"]["batch_size"], shuffle=True)
        model = BaseModel(config["model"]["name"])
        weights = compute_class_weights(train_labels)
        model = train_model(model, tokenizer, train_loader, num_epochs=config["training"]["epochs"], device=device, class_weights=weights)
        
        # Test on Apparel (Completely unseen)
        test_texts, test_labels, test_d_ids = load_amazon_split("english", "apparel", "test", max_samples=config["scenarios"]["max_samples_test"])
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, tokenizer, batch_size=config["training"]["batch_size"])
        res_s3 = evaluate_model(model, test_loader, device, "S3_UnseenDomain_Apparel")
        save_results(res_s3, "results/results_s3.json")

    # 4. SCENARIO 4: Unsupervised Domain Adaptation (DANN)
    if args.s in [0, 4]:
        print_banner("Scenario 4: DANN Adaptation (Amazon -> Twitter)")
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        # Source: Amazon Books
        s_texts, s_labels, s_d_ids = load_amazon_split("english", "books", "train", max_samples=config["scenarios"]["max_samples_train"])
        # Target: Twitter Unlabeled
        t_texts, t_labels, t_d_ids = load_tweeteval("train", max_samples=config["scenarios"]["max_samples_train"], unlabeled=True)
        
        s_loader = make_dataloader(s_texts, s_labels, s_d_ids, tokenizer, batch_size=config["training"]["batch_size"], shuffle=True)
        t_loader = make_dataloader(t_texts, t_labels, t_d_ids, tokenizer, batch_size=config["training"]["batch_size"], shuffle=True)
        
        model = DANNModel(config["model"]["name"])
        weights = compute_class_weights(s_labels)
        model = train_dann(model, tokenizer, s_loader, t_loader, num_epochs=int(config["training"]["epochs"]), lr=float(config["training"]["learning_rate"]), device=device, class_weights=weights)
        
        # Test on Twitter
        test_texts, test_labels, test_d_ids = load_tweeteval("test", max_samples=config["scenarios"]["max_samples_test"])
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, tokenizer, batch_size=config["training"]["batch_size"])
        res_s4 = evaluate_model(model, test_loader, device, "S4_DANN_Twitter")
        save_results(res_s4, "results/results_s4.json")
        
        # Vẽ t-SNE để xem DANN có kéo 2 miền Amazon và Twitter lại gần nhau không
        print("\n[Visualization] Drawing t-SNE for S4...")
        s_loader_small = make_dataloader(s_texts[:300], s_labels[:300], s_d_ids[:300], tokenizer, batch_size=16)
        t_loader_small = make_dataloader(t_texts[:300], t_labels[:300], t_d_ids[:300], tokenizer, batch_size=16)
        visualize_tsne(model, tokenizer, [s_loader_small, t_loader_small], ["Amazon (Source)", "Twitter (Target)"], device, "S4_DANN_Adaptation_Alignment")

    # 5. SCENARIO 5: Multilingual Joint Learning
    if args.s in [0, 5]:
        print_banner("Scenario 5: Multilingual Joint Learning (EN + VI)")
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        # Combine Amazon EN and VSFC VI
        t_en, l_en, d_en = load_amazon_split("english", "books", "train", max_samples=config["scenarios"]["max_samples_train"]//2)
        t_vi, l_vi, d_vi = load_vsfc("train", max_samples=config["scenarios"]["max_samples_train"]//2)
        
        train_texts = t_en + t_vi
        train_labels = l_en + l_vi
        train_d_ids = d_en + d_vi
        
        train_loader = make_dataloader(train_texts, train_labels, train_d_ids, tokenizer, batch_size=config["training"]["batch_size"], shuffle=True)
        model = BaseModel(config["model"]["name"])
        weights = compute_class_weights(train_labels)
        model = train_model(model, tokenizer, train_loader, num_epochs=config["training"]["epochs"], device=device, class_weights=weights)
        
        # Test on both
        print("\nTesting on English...")
        test_en_t, test_en_l, test_en_d = load_amazon_split("english", "books", "test", max_samples=500)
        test_loader_en = make_dataloader(test_en_t, test_en_l, test_en_d, tokenizer, batch_size=config["training"]["batch_size"])
        evaluate_model(model, test_loader_en, device, "S5_Joint_EN")
        
        print("\nTesting on Vietnamese...")
        test_vi_t, test_vi_l, test_vi_d = load_vsfc("test", max_samples=500)
        test_loader_vi = make_dataloader(test_vi_t, test_vi_l, test_vi_d, tokenizer, batch_size=config["training"]["batch_size"])
        evaluate_model(model, test_loader_vi, device, "S5_Joint_VI")

if __name__ == "__main__":
    main()
