# Train Detector Model

A **train detection & OCR chatbot**. It uses a custom-trained **YOLO11 segmentation**
model to find and outline trains in photos, and **PaddleOCR** to read any text/numbers
printed on them (e.g. locomotive IDs). A Gradio web app ties them together with a
human-in-the-loop learning loop.

## Features

- 🚆 **Detect & segment trains** with a YOLO11n-seg model trained on a custom dataset.
- 🔤 **Read text** on the image with PaddleOCR.
- 💬 **Chatbot UI** (Gradio): asks your name, takes an image, reports results.
- 🧠 **Human-in-the-loop learning**: confirm a detection (yes/no) and the image is
  added to the dataset — "yes" as a train sample, "no" as a background/negative
  sample — then retrain from the app to improve the model over time.

## Project layout

| Path | What it is |
|------|------------|
| `chatbot.py` | The Gradio web chatbot (detection + OCR + learning). |
| `prepare_dataset.py` | Converts CVAT polygon annotations (`annotations.xml`) to YOLO format + train/val split. |
| `train.py` | Trains the YOLO segmentation model. |
| `predict.py` | Runs the trained model on an image. |
| `ocr_test.py`, `ocr_pdf.py` | PaddleOCR examples (image + PDF). |
| `annotations.xml` | Source polygon annotations (CVAT 1.1 format). |
| `data.yaml` | YOLO dataset config. |
| `dataset/` | YOLO-format images + labels (train/val). |
| `train_images/` | Raw source images. |
| `runs/` | Training outputs, including the trained weights. |

The trained model lives at `runs/segment/runs/train_seg/weights/best.pt`.

## Setup

> **Important:** PaddlePaddle has no wheels for Python 3.14. Use **Python 3.12**.

```bash
# create a virtual environment with Python 3.12
py -3.12 -m venv paddle_env

# install dependencies
paddle_env\Scripts\python.exe -m pip install paddlepaddle paddleocr ultralytics gradio
```

## Usage

**Run the chatbot** (must use the virtual-environment Python):

```bash
paddle_env\Scripts\python.exe chatbot.py
```

Then open <http://127.0.0.1:7860>.

**Retrain from scratch:**

```bash
paddle_env\Scripts\python.exe prepare_dataset.py   # build dataset from annotations.xml
paddle_env\Scripts\python.exe train.py             # train the model
```

## Notes

- Runs on CPU by default. A CUDA-capable GPU makes training much faster.
- PaddleOCR must be created with `enable_mkldnn=False` on this setup to avoid a
  oneDNN inference crash.
- Run only **one** instance of the chatbot at a time — each one loads several heavy
  models, and multiple copies can exhaust memory.
