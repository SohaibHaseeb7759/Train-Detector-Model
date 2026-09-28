"""Run PaddleOCR on a PDF and dump recognized text + positions to JSON."""
import json
import sys
from paddleocr import PaddleOCR

pdf_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\DELL\Downloads\RocketMSL-blog-publishing-schedule.pdf"

ocr = PaddleOCR(use_textline_orientation=True, lang="en", enable_mkldnn=False)
results = ocr.predict(pdf_path)

pages = []
for page_idx, res in enumerate(results):
    texts = res.get("rec_texts", [])
    scores = res.get("rec_scores", [])
    polys = res.get("rec_polys", res.get("dt_polys", []))
    items = []
    for i, t in enumerate(texts):
        poly = polys[i] if i < len(polys) else None
        # poly is a list of [x,y] points; compute a simple bbox
        if poly is not None:
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            box = [min(xs), min(ys), max(xs), max(ys)]
        else:
            box = None
        items.append({
            "text": t,
            "score": float(scores[i]) if i < len(scores) else None,
            "box": box,  # [x0, y0, x1, y1]
        })
    pages.append({"page": page_idx + 1, "items": items})

out = "ocr_pdf_raw.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(pages, f, indent=2, ensure_ascii=False)

total = sum(len(p["items"]) for p in pages)
print(f"Pages: {len(pages)} | Text lines detected: {total}")
print(f"Raw results saved to {out}")
