import os
from PyPDF2 import PdfFileReader
from pdfminer.high_level import extract_text

# specify your source and destination directories
src_dir = 'documents'
dst_dir = 'documents1'

# create destination directory if it doesn't exist
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# iterate over all files in the source directory
for filename in os.listdir(src_dir):
    # check if the file is a PDF
    if filename.endswith('.pdf'):
        # construct full file path
        src_file = os.path.join(src_dir, filename)
        # extract text from the PDF file
        text = extract_text(src_file)
        # construct text file path
        txt_file = os.path.join(dst_dir, f'{os.path.splitext(filename)[0]}.txt')
        # save text to a file
        with open(txt_file, 'w', encoding ='utf-8') as f:
            f.write(text)
