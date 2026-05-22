import sys
import os

try:
    import pypdf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
    import pypdf

def extract_pdf_text(pdf_path, txt_path):
    print(f"Extracting {pdf_path} -> {txt_path}")
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            text += f"--- Page {i+1} ---\n"
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Done! Saved {len(reader.pages)} pages to {txt_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    pdf4 = r"d:\XuanDuc\TaiLieuKi8\CuoiKiCd4\Chapter4_MultidomainSA.pdf"
    pdf5 = r"d:\XuanDuc\TaiLieuKi8\CuoiKiCd4\Chapter5_MultilingualSA.pdf"
    
    extract_pdf_text(pdf4, r"d:\XuanDuc\TaiLieuKi8\CuoiKiCd4\project\backend\scratch\Chapter4_text.txt")
    extract_pdf_text(pdf5, r"d:\XuanDuc\TaiLieuKi8\CuoiKiCd4\project\backend\scratch\Chapter5_text.txt")
