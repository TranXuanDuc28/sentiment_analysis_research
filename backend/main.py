import argparse
import yaml
import os
import sys
import torch
import gc

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from src.dataset import load_amazon_split, load_vsfc, load_yelp, load_imdb, make_dataloader
from src.model import BaseModel, DANNModel, AdvancedMultiTaskModel
from src.train import train_model, train_dann, train_multitask, compute_class_weights
from src.evaluate import evaluate_model
from src.visualize_embeddings import visualize_tsne
from src.report_generator import generate_aggregate_report
from src.demo_plotter import generate_demo_plot
from src.utils import print_banner, save_results, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=str, default="0", help="Scenario to run (0=all, 1-12)")
    args = parser.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    BASE_TEST = config["scenarios"].get("base_samples_test", 2000)
    BASE_TRAIN = config["scenarios"].get("base_samples_train", 5000)
    FEW_SHOT = config["scenarios"].get("few_shot_samples", 500)
    
    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = int(config["training"]["epochs"])
    LR = float(config["training"]["learning_rate"])
    
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    # =========================================================================
    # CHẶNG 1: THÁCH THỨC ĐA MIỀN (MULTIDOMAIN ANALYSIS)
    # =========================================================================

    # --- S1: Multidomain Baseline (IMDb + Yelp -> Amazon) ---
    if args.s in ["0", "1"]:
        print_banner("Scenario 1: Multidomain Baseline (IMDb + Yelp -> Amazon)")
        t1, l1, d1, la1 = load_imdb("train", max_samples=BASE_TRAIN)
        t2, l2, d2, la2 = load_yelp("train", max_samples=BASE_TRAIN)
        t_all, l_all, d_all, la_all = t1 + t2, l1 + l2, d1 + d2, la1 + la2
        t_tr, t_vl, l_tr, l_vl, d_tr, d_vl, la_tr, la_vl = train_test_split(t_all, l_all, d_all, la_all, test_size=0.2, random_state=42)
        tr_ld = make_dataloader(t_tr, l_tr, d_tr, la_tr, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        vl_ld = make_dataloader(t_vl, l_vl, d_vl, la_vl, tokenizer, batch_size=BATCH_SIZE)
        model = BaseModel(config["model"]["name"])
        if not os.path.exists("checkpoints/model_s1_multidomain.pt"):
            w = compute_class_weights(l_tr)
            model = train_model(model, tokenizer, tr_ld, vl_ld, EPOCHS, LR, device, w)
            torch.save(model.state_dict(), "checkpoints/model_s1_multidomain.pt")
        else: model.load_state_dict(torch.load("checkpoints/model_s1_multidomain.pt", device))
        tt, tl, td, tla = load_amazon_split("english", "all", "test", BASE_TEST)
        save_results(evaluate_model(model, make_dataloader(tt, tl, td, tla, tokenizer, BATCH_SIZE), device, "S1_MD_Baseline"), "results/results_s1.json")

    # --- S2: Multidomain Multi-task Learning ---
    if args.s in ["0", "2"]:
        print_banner("Scenario 2: Multidomain Multi-task Learning (IMDb + Yelp)")
        t1, l1, d1, la1 = load_imdb("train", BASE_TRAIN)
        t2, l2, d2, la2 = load_yelp("train", BASE_TRAIN)
        t_all, l_all, d_all, la_all = t1+t2, l1+l2, d1+d2, la1+la2
        s_tr, s_vl, l_tr, l_vl, d_tr, d_vl, la_tr, la_vl = train_test_split(t_all, l_all, d_all, la_all, test_size=0.1, random_state=42)
        tr_ld = make_dataloader(s_tr, l_tr, d_tr, la_tr, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        vl_ld = make_dataloader(s_vl, l_vl, d_vl, la_vl, tokenizer, batch_size=BATCH_SIZE)
        model_mt = AdvancedMultiTaskModel(config["model"]["name"])
        if not os.path.exists("checkpoints/model_s2_mtl.pt"):
            w = compute_class_weights([lbl for lbl in l_tr if lbl >= 0])
            model_mt = train_multitask(model_mt, tokenizer, tr_ld, vl_ld, EPOCHS, LR/10.0, device, w)
            torch.save(model_mt.state_dict(), "checkpoints/model_s2_mtl.pt")
        else: model_mt.load_state_dict(torch.load("checkpoints/model_s2_mtl.pt", device))
        tt, tl, td, tla = load_amazon_split("english", "all", "test", BASE_TEST)
        save_results(evaluate_model(model_mt, make_dataloader(tt, tl, td, tla, tokenizer, BATCH_SIZE), device, "S2_MD_MTL"), "results/results_s2.json")

    # --- S3: Multidomain Adversarial Adaptation (DANN) ---
    if args.s in ["0", "3"]:
        print_banner("Scenario 3: Multidomain DANN (IMDb+Yelp -> Amazon)")
        t1, l1, d1, la1 = load_imdb("train", BASE_TRAIN)
        t2, l2, d2, la2 = load_yelp("train", BASE_TRAIN)
        s_texts, s_labels, s_d_ids, s_la_ids = t1+t2, l1+l2, d1+d2, la1+la2
        t_texts, t_labels, t_d_ids, t_la_ids = load_amazon_split("english", "all", "train", BASE_TRAIN*2, unlabeled=True)
        s_tr, s_vl, l_tr, l_vl, d_tr, d_vl, la_tr, la_vl = train_test_split(s_texts, s_labels, s_d_ids, s_la_ids, test_size=0.2, random_state=42)
        s_ld = make_dataloader(s_tr, l_tr, d_tr, la_tr, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        v_ld = make_dataloader(s_vl, l_vl, d_vl, la_vl, tokenizer, batch_size=BATCH_SIZE)
        t_ld = make_dataloader(t_texts, t_labels, t_d_ids, t_la_ids, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        model_dn = DANNModel(config["model"]["name"])
        if not os.path.exists("checkpoints/model_s3_dann.pt"):
            if os.path.exists("checkpoints/model_s1_multidomain.pt"):
                model_dn.load_state_dict(torch.load("checkpoints/model_s1_multidomain.pt", device), strict=False)
            w = compute_class_weights(l_tr)
            model_dn = train_dann(model_dn, tokenizer, s_ld, t_ld, v_ld, EPOCHS, LR/10.0, device, w)
            torch.save(model_dn.state_dict(), "checkpoints/model_s3_dann.pt")
        else: model_dn.load_state_dict(torch.load("checkpoints/model_s3_dann.pt", device))
        test_texts, test_labels, test_d_ids, test_la_ids = load_amazon_split("english", "all", "test", max_samples=BASE_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s3 = evaluate_model(model_dn, test_loader, device, "S3_Multidomain_DANN")
        save_results(res_s3, "results/results_s3.json")
        
        # Vẽ t-SNE cho S3
        visualize_tsne(model_dn, tokenizer, [s_ld, t_ld], ["Source (IMDb+Yelp)", "Target (Amazon)"], device, "S3_Multidomain_Alignment")

    # =========================================================================
    # CHẶNG 2: THÁCH THỨC ĐA NGÔN NGỮ (MULTILINGUAL ANALYSIS)
    # =========================================================================

    model_src_path = "checkpoints/model_src_en_fr.pt"
    model_src = BaseModel(config["model"]["name"])
    if not os.path.exists(model_src_path):
        print_banner("Training Base Multi-source Model (IMDb[EN] + Amazon[FR])")
        t1, l1, d1, la1 = load_imdb("train", BASE_TRAIN)
        t2, l2, d2, la2 = load_amazon_split("french", "all", "train", BASE_TRAIN)
        t_all, l_all, d_all, la_all = t1+t2, l1+l2, d1+d2, la1+la2
        t_tr, t_vl, l_tr, l_vl, d_tr, d_vl, la_tr, la_vl = train_test_split(t_all, l_all, d_all, la_all, test_size=0.2, random_state=42)
        ld_tr = make_dataloader(t_tr, l_tr, d_tr, la_tr, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        ld_vl = make_dataloader(t_vl, l_vl, d_vl, la_vl, tokenizer, batch_size=BATCH_SIZE)
        model_src = train_model(model_src, tokenizer, ld_tr, ld_vl, EPOCHS, LR, device, compute_class_weights(l_tr))
        torch.save(model_src.state_dict(), model_src_path)
    else: model_src.load_state_dict(torch.load(model_src_path, device))

    # --- S4: Multilingual Zero-shot (Anh + Pháp -> Việt) ---
    if args.s in ["0", "4"]:
        print_banner("Scenario 4: Multilingual Zero-shot (EN+FR -> VI)")
        tt, tl, td, tla = load_vsfc("test", BASE_TEST)
        save_results(evaluate_model(model_src, make_dataloader(tt, tl, td, tla, tokenizer, BATCH_SIZE), device, "S4_ML_ZeroShot"), "results/results_s4.json")

    # --- S5: Translation-Based Baseline (VI -> EN) ---
    if args.s in ["0", "5"]:
        print_banner("Scenario 5: Translation-Based Baseline (VI -> EN)")
        tt, tl, td, tla = load_vsfc("test", BASE_TEST)
        try:
            from deep_translator import GoogleTranslator
            from tqdm import tqdm
            texts_en = [GoogleTranslator(source='vi', target='en').translate(t[:4999]) for t in tqdm(tt)]
        except: texts_en = tt
        save_results(evaluate_model(model_src, make_dataloader(texts_en, tl, td, tla, tokenizer, BATCH_SIZE), device, "S5_ML_Translation"), "results/results_s5.json")

    # --- S6: Triple-Language Joint Training ---
    if args.s in ["0", "6"]:
        print_banner("Scenario 6: Joint Training (EN+FR+VI)")
        t1, l1, d1, la1 = load_imdb("train", BASE_TRAIN)
        t2, l2, d2, la2 = load_amazon_split("french", "all", "train", BASE_TRAIN)
        t3, l3, d3, la3 = load_vsfc("train", BASE_TRAIN)
        t_all, l_all, d_all, la_all = t1+t2+t3, l1+l2+l3, d1+d2+d3, la1+la2+la3
        t_tr, t_vl, l_tr, l_vl, d_tr, d_vl, la_tr, la_vl = train_test_split(t_all, l_all, d_all, la_all, test_size=0.2, random_state=42)
        model = BaseModel(config["model"]["name"])
        if not os.path.exists("checkpoints/model_s6_joint.pt"):
            model = train_model(model, tokenizer, make_dataloader(t_tr, l_tr, d_tr, la_tr, tokenizer, BATCH_SIZE, True), make_dataloader(t_vl, l_vl, d_vl, la_vl, tokenizer, BATCH_SIZE), EPOCHS, LR, device, compute_class_weights(l_tr))
            torch.save(model.state_dict(), "checkpoints/model_s6_joint.pt")
        else: model.load_state_dict(torch.load("checkpoints/model_s6_joint.pt", device))
        tt, tl, td, tla = load_vsfc("test", BASE_TEST)
        save_results(evaluate_model(model, make_dataloader(tt, tl, td, tla, tokenizer, BATCH_SIZE), device, "S6_ML_Joint"), "results/results_s6.json")

    # =========================================================================
    # CHẶNG 3: GIẢI PHÁP HỢP NHẤT (UNIFIED FRAMEWORK)
    # =========================================================================

    # --- S7: Unified Zero-shot ---
    if args.s in ["0", "7"]:
        print_banner("Scenario 7: Unified Zero-shot")
        model = BaseModel(config["model"]["name"])
        if os.path.exists("checkpoints/model_s1_multidomain.pt"):
            model.load_state_dict(torch.load("checkpoints/model_s1_multidomain.pt", device))
        tt, tl, td, tla = load_vsfc("test", BASE_TEST)
        save_results(evaluate_model(model, make_dataloader(tt, tl, td, tla, tokenizer, BATCH_SIZE), device, "S7_UN_ZeroShot"), "results/results_s7.json")

    # --- S8: Unified Adversarial Adaptation (DANN) ---
    if args.s in ["0", "8"]:
        print_banner("Scenario 8: Unified DANN")
        t1, l1, d1, la1 = load_imdb("train", BASE_TRAIN)
        t2, l2, d2, la2 = load_yelp("train", BASE_TRAIN)
        t3, l3, d3, la3 = load_amazon_split("french", "all", "train", BASE_TRAIN)
        s_texts, s_labels, s_d_ids, s_la_ids = t1+t2+t3, l1+l2+l3, d1+d2+d3, la1+la2+la3
        t_vi, l_vi, d_vi, la_vi = load_vsfc("train", BASE_TRAIN*3, unlabeled=True)
        s_tr, s_vl, l_tr, l_vl, d_tr, d_vl, la_tr, la_vl = train_test_split(s_texts, s_labels, s_d_ids, s_la_ids, test_size=0.2, random_state=42)
        s_ld = make_dataloader(s_tr, l_tr, d_tr, la_tr, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        t_ld = make_dataloader(t_vi, l_vi, d_vi, la_vi, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        model_dn = DANNModel(config["model"]["name"])
        if not os.path.exists("checkpoints/model_s8_unified_dann.pt"):
            model_dn = train_dann(model_dn, tokenizer, s_ld, t_ld, make_dataloader(s_vl, l_vl, d_vl, la_vl, tokenizer, batch_size=BATCH_SIZE, shuffle=False), EPOCHS, LR/10.0, device, compute_class_weights(l_tr))
            torch.save(model_dn.state_dict(), "checkpoints/model_s8_unified_dann.pt")
        else: model_dn.load_state_dict(torch.load("checkpoints/model_s8_unified_dann.pt", device))
        test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", BASE_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi, tokenizer, BATCH_SIZE)
        save_results(evaluate_model(model_dn, test_loader_vi, device, "S8_UN_DANN"), "results/results_s8.json")
        
        # Vẽ t-SNE cho S8
        visualize_tsne(model_dn, tokenizer, [s_ld, t_ld], ["Source (EN+FR+Yelp)", "Target (Vietnamese)"], device, "S8_Unified_Alignment")

    # --- S9: Unified Multi-task Framework ---
    if args.s in ["0", "9"]:
        print_banner("Scenario 9: Unified Multi-task Framework")
        t1, l1, d1, la1 = load_imdb("train", BASE_TRAIN)
        t2, l2, d2, la2 = load_yelp("train", BASE_TRAIN)
        t3, l3, d3, la3 = load_amazon_split("english", "all", "train", BASE_TRAIN)
        t_vi, l_vi, d_vi, la_vi = load_vsfc("train", BASE_TRAIN, unlabeled=True)
        t_all = t1+t2+t3+t_vi
        l_all = l1+l2+l3+l_vi
        d_all = d1+d2+d3+d_vi
        la_all = la1+la2+la3+la_vi
        s_tr, s_vl, l_tr, l_vl, d_tr, d_vl, la_tr, la_vl = train_test_split(t_all, l_all, d_all, la_all, test_size=0.1, random_state=42)
        model_mt = AdvancedMultiTaskModel(config["model"]["name"])
        if not os.path.exists("checkpoints/model_s9_multitask.pt"):
            model_mt = train_multitask(model_mt, tokenizer, make_dataloader(s_tr, l_tr, d_tr, la_tr, tokenizer, BATCH_SIZE, True), make_dataloader(s_vl, l_vl, d_vl, la_vl, tokenizer, BATCH_SIZE), EPOCHS, LR/10.0, device, compute_class_weights([lbl for lbl in l_tr if lbl >= 0]))
            torch.save(model_mt.state_dict(), "checkpoints/model_s9_multitask.pt")
        else: model_mt.load_state_dict(torch.load("checkpoints/model_s9_multitask.pt", device))
        tt, tl, td, tla = load_vsfc("test", BASE_TEST)
        save_results(evaluate_model(model_mt, make_dataloader(tt, tl, td, tla, tokenizer, BATCH_SIZE), device, "S9_UN_MultiTask"), "results/results_s9.json")

    # =========================================================================
    # CHẶNG 4: ĐỐI SÁNH MÔ HÌNH (MODEL ABLATION)
    # =========================================================================

    if args.s in ["0", "10", "11", "12"]:
        mbert_name = "bert-base-multilingual-cased"
        tokenizer_mb = AutoTokenizer.from_pretrained(mbert_name)
        print_banner("Cluster 4: Model Comparison (mBERT vs XLM-R)")

        # S10: mBERT on MD DANN (Compare with S3)
        if args.s in ["0", "10"]:
            print_banner("Scenario 10: mBERT on MD DANN")
            t1, l1, d1, la1 = load_imdb("train", BASE_TRAIN)
            t2, l2, d2, la2 = load_yelp("train", BASE_TRAIN)
            s_tr, s_vl, l_tr, l_vl, d_tr, d_vl, la_tr, la_vl = train_test_split(t1+t2, l1+l2, d1+d2, la1+la2, test_size=0.2, random_state=42)
            t_texts, t_labels, t_d_ids, t_la_ids = load_amazon_split("english", "all", "train", BASE_TRAIN*2, unlabeled=True)
            model_mb = DANNModel(mbert_name)
            if not os.path.exists("checkpoints/model_s10_mbert_dann.pt"):
                model_mb = train_dann(model_mb, tokenizer_mb, make_dataloader(s_tr, l_tr, d_tr, la_tr, tokenizer_mb, BATCH_SIZE, True), make_dataloader(t_texts, t_labels, t_d_ids, t_la_ids, tokenizer_mb, BATCH_SIZE, True), make_dataloader(s_vl, l_vl, d_vl, la_vl, tokenizer_mb, BATCH_SIZE), EPOCHS, LR/10.0, device, compute_class_weights(l_tr))
                torch.save(model_mb.state_dict(), "checkpoints/model_s10_mbert_dann.pt")
            else: model_mb.load_state_dict(torch.load("checkpoints/model_s10_mbert_dann.pt", device))
            tt, tl, td, tla = load_amazon_split("english", "all", "test", BASE_TEST)
            save_results(evaluate_model(model_mb, make_dataloader(tt, tl, td, tla, tokenizer_mb, BATCH_SIZE), device, "S10_mBERT_MD"), "results/results_s10.json")

        # S11: mBERT on ML Zero-shot (Compare with S4)
        if args.s in ["0", "11"]:
            print_banner("Scenario 11: mBERT on ML Zero-shot")
            t1, l1, d1, la1 = load_imdb("train", BASE_TRAIN)
            t2, l2, d2, la2 = load_amazon_split("french", "all", "train", BASE_TRAIN)
            t_tr, t_vl, l_tr, l_vl, d_tr, d_vl, la_tr, la_vl = train_test_split(t1+t2, l1+l2, d1+d2, la1+la2, test_size=0.2, random_state=42)
            model_mb = BaseModel(mbert_name)
            if not os.path.exists("checkpoints/model_s11_mbert_src.pt"):
                model_mb = train_model(model_mb, tokenizer_mb, make_dataloader(t_tr, l_tr, d_tr, la_tr, tokenizer_mb, BATCH_SIZE, True), make_dataloader(t_vl, l_vl, d_vl, la_vl, tokenizer_mb, BATCH_SIZE), EPOCHS, LR, device, compute_class_weights(l_tr))
                torch.save(model_mb.state_dict(), "checkpoints/model_s11_mbert_src.pt")
            else: model_mb.load_state_dict(torch.load("checkpoints/model_s11_mbert_src.pt", device))
            tt, tl, td, tla = load_vsfc("test", BASE_TEST)
            save_results(evaluate_model(model_mb, make_dataloader(tt, tl, td, tla, tokenizer_mb, BATCH_SIZE), device, "S11_mBERT_ML"), "results/results_s11.json")

        # S12: mBERT on UN DANN (Compare with S8)
        if args.s in ["0", "12"]:
            print_banner("Scenario 12: mBERT on Unified DANN")
            t1, l1, d1, la1 = load_imdb("train", BASE_TRAIN)
            t2, l2, d2, la2 = load_yelp("train", BASE_TRAIN)
            t3, l3, d3, la3 = load_amazon_split("french", "all", "train", BASE_TRAIN)
            s_tr, s_vl, l_tr, l_vl, d_tr, d_vl, la_tr, la_vl = train_test_split(t1+t2+t3, l1+l2+l3, d1+d2+d3, la1+la2+la3, test_size=0.2, random_state=42)
            t_vi, l_vi, d_vi, la_vi = load_vsfc("train", BASE_TRAIN*3, unlabeled=True)
            s_ld = make_dataloader(s_tr, l_tr, d_tr, la_tr, tokenizer_mb, batch_size=BATCH_SIZE, shuffle=True)
            t_ld = make_dataloader(t_vi, l_vi, d_vi, la_vi, tokenizer_mb, batch_size=BATCH_SIZE, shuffle=True)
            model_mb = DANNModel(mbert_name)
            if not os.path.exists("checkpoints/model_s12_mbert_unified.pt"):
                model_mb = train_dann(model_mb, tokenizer_mb, s_ld, t_ld, make_dataloader(s_vl, l_vl, d_vl, la_vl, tokenizer_mb, BATCH_SIZE), EPOCHS, LR/10.0, device, compute_class_weights(l_tr))
                torch.save(model_mb.state_dict(), "checkpoints/model_s12_mbert_unified.pt")
            else: model_mb.load_state_dict(torch.load("checkpoints/model_s12_mbert_unified.pt", device))
            tt, tl, td, tla = load_vsfc("test", BASE_TEST)
            ld = make_dataloader(tt, tl, td, tla, tokenizer_mb, BATCH_SIZE)
            save_results(evaluate_model(model_mb, ld, device, "S12_mBERT_UN"), "results/results_s12.json")
            
            # Vẽ t-SNE cho S12 (để so sánh với S8)
            visualize_tsne(model_mb, tokenizer_mb, [s_ld, t_ld], ["Source (mBERT)", "Target (Vietnamese)"], device, "S12_mBERT_Unified_Alignment")

    print_banner("ALL EXPERIMENTS COMPLETED")
    try: 
        generate_aggregate_report()
        generate_demo_plot()
    except Exception as e: print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
