"""Quick smoke test that PaddleOCR is installed and working."""
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

# 1. Make a simple test image with known text.
img = Image.new("RGB", (600, 200), "white")
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("arial.ttf", 48)
except Exception:
    font = ImageFont.load_default()
draw.text((30, 40), "Hello PaddleOCR 123", fill="black", font=font)
draw.text((30, 110), "Train Model Test", fill="black", font=font)
test_path = "ocr_sample.png"
img.save(test_path)

# 2. Run OCR (first run downloads the models automatically).
ocr = PaddleOCR(use_textline_orientation=True, lang="en", enable_mkldnn=False)
result = ocr.predict(test_path)

# 3. Print recognized text.
print("\n================ OCR RESULT ================")
for res in result:
    texts = res.get("rec_texts", [])
    scores = res.get("rec_scores", [])
    for t, s in zip(texts, scores):
        print(f"  '{t}'   (confidence {s:.2%})")
print("============================================")
