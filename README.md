# Train Detector Model

A **train detection & OCR app**. It uses a custom-trained **YOLO11 segmentation**
model to find and outline trains in photos, and **PaddleOCR** to read text/numbers
(e.g. locomotive IDs). A **Gradio** web app ties them together, with a human-in-the-loop
learning loop and a video-to-text (OCR → JSON) tool.

## Features

- 🚆 **Detect & segment trains** with a YOLO11n-seg model trained on a custom dataset.
- 🔤 **Read text** on an image with PaddleOCR.
- 💬 **Chatbot UI** (Gradio): asks your name, takes an image, reports results.
- 🧠 **Human-in-the-loop learning**: confirm a detection (yes/no) and the image is
  added to the dataset — "yes" as a train sample, "no" as a background/negative
  sample — then retrain from the app to improve the model over time.
- 🎞️ **Video OCR → JSON**: upload a video, scan its frames, and download every piece
  of text found as JSON (with frame numbers, timestamps, and confidences).

---

## Quick start (for a new user)

### 1. Prerequisites

- **Windows** (commands below are for PowerShell)
- **Python 3.12** — install from <https://www.python.org/downloads/> or with
  `winget install Python.Python.3.12`
  > ⚠️ **Do not use Python 3.13 or 3.14** — PaddlePaddle has no wheels for them and
  > the install will fail. This project needs **3.12**.
- **Git**

Check Python 3.12 is available:

```bash
py -3.12 --version
```

### 2. Clone the repository

```bash
git clone https://github.com/SohaibHaseeb7759/Train-Detector-Model.git
cd Train-Detector-Model
```

### 3. Create the virtual environment (Python 3.12)

```bash
py -3.12 -m venv paddle_env
```

### 4. Install the dependencies

```bash
paddle_env\Scripts\python.exe -m pip install --upgrade pip
paddle_env\Scripts\python.exe -m pip install paddlepaddle paddleocr ultralytics gradio opencv-python
```

> This downloads ~2 GB and takes a few minutes. The OCR models themselves download
> automatically the first time you run the app.

### 5. Run the app

```bash
paddle_env\Scripts\python.exe chatbot.py
```

Wait ~20–30 seconds for the models to load (you'll see `Models ready.`), then open:

👉 **<http://127.0.0.1:7860>**

To stop the app, press **Ctrl + C** in the terminal.

---

## How to use

The app has two tabs:

### 💬 Chatbot (image)
1. Type your **name**.
2. Upload an **image** using the 📎 clip in the message box.
3. The bot detects trains, reads any text, and asks you to confirm.
   - **yes** → saved as a train training sample.
   - **no** → saved as a "not a train" (negative) sample.
4. Click **🔁 Retrain now** to fold confirmed images into the model (runs in the
   background; takes a while on CPU).

### 🎞️ Video OCR → JSON
1. Upload a **video**.
2. Set **frames per second to scan** and **max frames** (safety cap).
3. Click **🔍 Scan video for text**.
4. View the JSON on screen and **download** the `.json` file.

---

## Retrain the model from scratch (optional)

```bash
paddle_env\Scripts\python.exe prepare_dataset.py   # build dataset from annotations.xml
paddle_env\Scripts\python.exe train.py             # train the YOLO model
```

The trained model is saved at `runs/segment/runs/train_seg/weights/best.pt`.

---

## Project layout

| Path | What it is |
|------|------------|
| `chatbot.py` | The Gradio web app (image chatbot + video OCR). |
| `prepare_dataset.py` | Converts CVAT polygon annotations (`annotations.xml`) to YOLO format + train/val split. |
| `train.py` | Trains the YOLO segmentation model. |
| `predict.py` | Runs the trained model on a single image. |
| `ocr_test.py`, `ocr_pdf.py` | PaddleOCR examples (image + PDF). |
| `annotations.xml` | Source polygon annotations (CVAT 1.1 format). |
| `data.yaml` | YOLO dataset config. |
| `dataset/` | YOLO-format images + labels (train/val). |
| `train_images/` | Raw source images. |
| `runs/` | Training outputs, including the trained weights. |

---

## Troubleshooting

- **`No matching distribution found for paddlepaddle`** → You're on Python 3.13/3.14.
  Recreate the venv with **Python 3.12**.
- **`cv2.error: bad allocation` / "paging file too small"** → Out of memory. Run only
  **one** instance of the app at a time. To clear stuck instances:
  ```bash
  Get-Process python | Stop-Process -Force
  ```
- **OCR crashes with a oneDNN error** → PaddleOCR must be created with
  `enable_mkldnn=False` (already set in `chatbot.py`).
- **`yolo` command not found** → use `paddle_env\Scripts\python.exe <script>.py`
  instead of the global `python`.

## Notes

- Runs on **CPU** by default. A CUDA-capable GPU makes training much faster.
- Everything must be run with the **virtual-environment Python**
  (`paddle_env\Scripts\python.exe`), not the system `python`.
