import fitz # PyMuPDF
import io
from PIL import Image
import os

pdf_file = r"c:\Users\ADMIN\Downloads\CIKD_STAGECD_TRANSFER\DL (1).pdf"
out_dir = r"c:\Users\ADMIN\Downloads\CIKD_STAGECD_TRANSFER\latex_project\figures"

os.makedirs(out_dir, exist_ok=True)
pdf = fitz.open(pdf_file)
image_count = 1
for i in range(len(pdf)):
    page = pdf[i]
    images = page.get_images(full=True)
    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = pdf.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image = Image.open(io.BytesIO(image_bytes))
        image.save(os.path.join(out_dir, f"figure_{image_count}.{image_ext}"))
        print(f"Extracted figure_{image_count}.{image_ext} from page {i+1}")
        image_count += 1
