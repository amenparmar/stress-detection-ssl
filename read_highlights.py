import pdfplumber
import fitz  # PyMuPDF

# Try PyMuPDF to get highlighted text
path = r'C:\Users\amenp\Desktop\Research Paper maker\AI Detector - QuillBot AI.pdf'
doc = fitz.open(path)

for page_num, page in enumerate(doc):
    annots = page.annots()
    highlights = []
    for annot in annots:
        if annot.type[0] == 8:  # Highlight annotation
            color = annot.colors.get('stroke') or annot.colors.get('fill')
            rect = annot.rect
            # Extract text in the highlighted area
            words = page.get_text("words", clip=rect)
            text = " ".join([w[4] for w in words])
            highlights.append((color, text[:300]))
    
    if highlights:
        print(f"\n=== PAGE {page_num+1} HIGHLIGHTS ===")
        for color, text in highlights:
            print(f"  COLOR={color}: {text}")

print("\n\nNow checking for colored/background text spans...")
for page_num, page in enumerate(doc):
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if block.get("type") == 0:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    color = span.get("color")
                    # Orange-ish colors
                    r = (color >> 16) & 0xFF
                    g = (color >> 8) & 0xFF
                    b = color & 0xFF
                    # Orange: high red, medium green, low blue
                    if r > 180 and g > 80 and g < 180 and b < 80:
                        print(f"PAGE {page_num+1} ORANGE r={r} g={g} b={b}: {span['text'][:200]}")
