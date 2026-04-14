import fitz
import os

path = r'C:\Users\amenp\Desktop\Research Paper maker\AI Detector - QuillBot AI.pdf'
doc = fitz.open(path)
out_dir = r'D:\scratch\stress_detection\pdf_pages'
os.makedirs(out_dir, exist_ok=True)

for i, page in enumerate(doc):
    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for clarity
    pix = page.get_pixmap(matrix=mat)
    out_path = os.path.join(out_dir, f'page_{i+1:02d}.png')
    pix.save(out_path)
    print(f"Saved page {i+1} -> {out_path}")

print("Done!")
