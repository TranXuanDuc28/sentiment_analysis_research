import pypdf
import sys

def extract_pdf_info(pdf_path, txt_file):
    txt_file.write(f"\n=========================================\n")
    txt_file.write(f"PDF: {pdf_path}\n")
    txt_file.write(f"=========================================\n")
    reader = pypdf.PdfReader(pdf_path)
    for i in range(len(reader.pages)):
        text = reader.pages[i].extract_text()
        if not text:
            continue
        txt_file.write(f"\n--- PAGE {i+1} ---\n")
        txt_file.write(text)

with open("backend/scratch/extracted_pdf_text.txt", "w", encoding="utf-8") as f:
    extract_pdf_info("d:\\XuanDuc\\TaiLieuKi8\\CuoiKiCd4\\Chapter4_MultidomainSA.pdf", f)
    extract_pdf_info("d:\\XuanDuc\\TaiLieuKi8\\CuoiKiCd4\\Chapter5_MultilingualSA.pdf", f)

print("Extraction completed!")
