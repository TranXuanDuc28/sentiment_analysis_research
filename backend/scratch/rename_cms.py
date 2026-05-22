import os
import shutil

plots_dir = r"d:\XuanDuc\TaiLieuKi8\CuoiKiCd4\project\backend\results\plots"

CM_RENAME_MAP = {
    "cm_1a_zeroshot_imdb_amazon.png": "cm_s1_singlesource_tl.png",
    "cm_s1_md_baseline.png": "cm_s2_md_baseline.png",
    "cm_s1b_md_fewshot.png": "cm_s3_md_fewshot.png",
    "cm_s2_md_mtl.png": "cm_s4_md_mtl.png",
    "cm_s3_multidomain_dann.png": "cm_s5_multidomain_dann.png",
    "cm_s0_baseline_vsfc.png": "cm_s6_monolingual_vi.png",
    "cm_s4_ml_zeroshot.png": "cm_s7_ml_zeroshot.png",
    "cm_s4b_ml_fewshot.png": "cm_s8_ml_fewshot.png",
    "cm_s5_ml_translation.png": "cm_s9_ml_translation.png",
    "cm_s6_ml_joint.png": "cm_s10_ml_joint.png",
    "cm_s7_un_zeroshot.png": "cm_s11_un_zeroshot.png",
    "cm_s8_un_dann.png": "cm_s12_un_dann.png",
    "cm_s9_un_multitask.png": "cm_s13_un_multitask.png",
    "cm_s10_mbert_md.png": "cm_s14_mbert_md.png",
    "cm_s11_mbert_ml.png": "cm_s15_mbert_ml.png",
    "cm_s12_mbert_un.png": "cm_s16_mbert_un.png",
}

print("[INFO] Renaming confusion matrix files...")
for src, dst in CM_RENAME_MAP.items():
    src_path = os.path.join(plots_dir, src)
    dst_path = os.path.join(plots_dir, dst)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f"[OK] Moved {src} -> {dst}")
    else:
        print(f"[WARN] {src} not found, skipping.")

print("[INFO] Renaming completed!")
