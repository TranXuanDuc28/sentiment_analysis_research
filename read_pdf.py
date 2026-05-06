import sys
try:
    import PyPDF2
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf2"])
    import PyPDF2

def read_pdf(file_path):
    try:
        reader = PyPDF2.PdfReader(file_path)
        text = ""
        for i in range(min(5, len(reader.pages))): # Read first 5 pages
            text += reader.pages[i].extract_text() + "\n"
        with open("pdf_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("Text extracted to pdf_text.txt")
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    read_pdf("C:/Users/MSI/Downloads/SentXFormer_a_transformer-enhanced_hybrid_deep_lea.pdf")
