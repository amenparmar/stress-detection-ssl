import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pdfplumber', '-q'])
import pdfplumber

path = r'C:\Users\amenp\Desktop\Research Paper maker\AI Detector - QuillBot AI.pdf'
with pdfplumber.open(path) as pdf:
    for i, page in enumerate(pdf.pages):
        print(f'=== PAGE {i+1} ===')
        words = page.extract_words(extra_attrs=['fontname','size','non_stroking_color','stroking_color'])
        for w in words:
            clr = w.get('non_stroking_color')
            txt = w['text']
            print(f'COLOR={clr} | {txt}')
