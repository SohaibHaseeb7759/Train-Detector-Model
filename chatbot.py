"""
Train Detection Chatbot (with human-in-the-loop learning)
---------------------------------------------------------
A Gradio web chatbot that:
  1. Asks for your name.
  2. Asks you to upload an image.
  3. Uses your trained YOLO model to detect/outline trains.
       - If NO train is found, it stops and says so (no OCR).
  4. If a train IS found, it reads text with PaddleOCR and asks you to
     confirm it really is a train.
       - If you say YES, the image + predicted outline are added to the
         training dataset so the model can learn from it.
  5. A "Retrain" button folds all confirmed images into the model in the
     background and hot-swaps to the improved model when done.

Run with the paddle_env Python:
    & ".\paddle_env\Scripts\python.exe" chatbot.py
"""
import os
import shutil
import threading
import time

import cv2
import gradio as gr
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WEIGHTS = r"runs\segment\runs\train_seg\weights\best.pt"
DATA_YAML = "data.yaml"
RESULT_IMG = "chatbot_result.jpg"
STAGING_DIR = "pending_samples"                 # confirmed-image holding area
TRAIN_IMG_DIR = os.path.join("dataset", "images", "train")
TRAIN_LBL_DIR = os.path.join("dataset", "labels", "train")
for d in (STAGING_DIR, TRAIN_IMG_DIR, TRAIN_LBL_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Load both models ONCE at startup (slow part happens here, not per message).
# ---------------------------------------------------------------------------
print("Loading YOLO model...")
YOLO_MODEL = YOLO(WEIGHTS)

print("Loading PaddleOCR model...")
# Load only the detection + recognition models to keep memory low. The doc
# orientation / unwarping / textline-orientation models are for scanned
# documents and are not needed for reading text off photos.
OCR_MODEL = PaddleOCR(
    lang="en",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
print("Models ready.")

# Shared training status (used by the background retrain thread).
TRAIN_STATUS = {"busy": False, "msg": ""}

YES_WORDS = {"yes", "y", "yeah", "yep", "yup", "correct", "right", "true", "ye", "sure"}
NO_WORDS = {"no", "n", "nope", "nah", "wrong", "incorrect", "false"}


# ---------------------------------------------------------------------------
# Core model helpers
# ---------------------------------------------------------------------------
def detect_trains(image_path):
    """Run YOLO. Returns (n_trains, confidences, label_lines, annotated_saved)."""
    results = YOLO_MODEL.predict(image_path, conf=0.25, save=False, verbose=False)
    r = results[0]
    n_trains = len(r.boxes) if r.boxes is not None else 0
    confidences = [float(b.conf[0]) for b in r.boxes] if n_trains else []

    # Save annotated image (masks + boxes drawn by YOLO). With no detections
    # this is effectively the original image. Guard against low-memory
    # allocation errors so a draw failure never breaks the whole upload.
    try:
        annotated = r.plot()
        cv2.imwrite(RESULT_IMG, annotated)
    except Exception as e:
        print(f"[warn] could not draw annotation ({e}); using original image")
        shutil.copy2(image_path, RESULT_IMG)

    # Build YOLO segmentation label lines from the predicted masks, so a
    # confirmed image can be added to the dataset as a training example.
    label_lines = []
    if n_trains and r.masks is not None:
        for cls_id, poly in zip(r.boxes.cls.tolist(), r.masks.xyn):
            coords = " ".join(f"{float(x):.6f} {float(y):.6f}" for x, y in poly)
            label_lines.append(f"{int(cls_id)} {coords}")

    return n_trains, confidences, label_lines


def read_text(image_path):
    """Run PaddleOCR and return a list of readable strings."""
    texts = []
    try:
        ocr_out = OCR_MODEL.predict(image_path)
        for res in ocr_out:
            for t, s in zip(res.get("rec_texts", []), res.get("rec_scores", [])):
                if s >= 0.5 and t.strip():
                    texts.append(t.strip())
    except Exception as e:
        texts = [f"(OCR error: {e})"]
    return texts


def save_confirmed_sample(staged_image_path, label_lines):
    """Copy a confirmed-train image + its predicted label into the dataset."""
    stem = f"user_{int(time.time() * 1000)}"
    img_dst = os.path.join(TRAIN_IMG_DIR, stem + ".jpg")
    lbl_dst = os.path.join(TRAIN_LBL_DIR, stem + ".txt")
    shutil.copy2(staged_image_path, img_dst)
    with open(lbl_dst, "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines))
    return img_dst


def save_negative_sample(staged_image_path):
    """Copy a 'not a train' image with an EMPTY label -> a background example.

    YOLO treats an image whose label file is empty as a negative sample, which
    teaches the model what is NOT a train and reduces false positives.
    """
    stem = f"neg_{int(time.time() * 1000)}"
    img_dst = os.path.join(TRAIN_IMG_DIR, stem + ".jpg")
    lbl_dst = os.path.join(TRAIN_LBL_DIR, stem + ".txt")
    shutil.copy2(staged_image_path, img_dst)
    open(lbl_dst, "w", encoding="utf-8").close()  # empty file = no objects
    return img_dst


def _retrain_worker(epochs):
    """Background thread: fine-tune from current weights, then hot-swap model."""
    global YOLO_MODEL
    try:
        TRAIN_STATUS["busy"] = True
        TRAIN_STATUS["msg"] = "🟠 Retraining in progress… (this takes a while on CPU)"
        m = YOLO(WEIGHTS)  # start from the current best weights
        m.train(
            data=DATA_YAML,
            epochs=epochs,
            imgsz=640,
            batch=8,
            device="cpu",
            project="runs",
            name="retrain",
            exist_ok=True,
            verbose=False,
        )
        new_weights = str(m.trainer.best)  # exact path to the new best.pt
        YOLO_MODEL = YOLO(new_weights)     # hot-swap the live model
        TRAIN_STATUS["msg"] = "✅ Retraining done — now using the improved model."
    except Exception as e:
        TRAIN_STATUS["msg"] = f"⚠️ Retraining failed: {e}"
    finally:
        TRAIN_STATUS["busy"] = False


# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------
def chat_fn(message, history, state):
    text = (message.get("text") or "").strip()
    files = message.get("files") or []

    user_display = text if text else "📎 (image uploaded)"
    history = history + [{"role": "user", "content": user_display}]

    stage = state["stage"]
    result_image = gr.update()

    # ---- Stage: ask for the name ----------------------------------------
    if stage == "ask_name":
        if not text:
            bot = "Please tell me your name first. 🙂"
        else:
            state["name"] = text.split()[0].capitalize()
            state["stage"] = "ask_image"
            bot = (f"Nice to meet you, **{state['name']}**! 👋\n\n"
                   "Now upload an image (📎 clip) and I'll check it for trains.")

    # ---- Stage: waiting for a yes/no confirmation -----------------------
    elif stage == "confirm":
        # If they upload a new image instead of answering, treat it as a new run.
        if files:
            state["stage"] = "ask_image"
            state["pending"] = None
            return chat_fn(message, history[:-1], state)

        low = text.lower()
        pending = state.get("pending")
        if low in YES_WORDS and pending:
            save_confirmed_sample(pending["image"], pending["labels"])
            state["new_samples"] = state.get("new_samples", 0) + 1
            state["pending"] = None
            state["stage"] = "ask_image"
            bot = (f"Great — thanks {state.get('name','')}! ✅ I've added this image to my "
                   f"training data (**{state['new_samples']}** new sample(s) this session).\n\n"
                   "Click **🔁 Retrain now** on the right whenever you want me to learn from "
                   "them, or just upload another image.")
        elif low in NO_WORDS:
            if pending:
                save_negative_sample(pending["image"])
                state["new_negatives"] = state.get("new_negatives", 0) + 1
            state["pending"] = None
            state["stage"] = "ask_image"
            bot = ("Thanks for the correction! 🙏 I've saved that as a "
                   f"**not-a-train** example (**{state.get('new_negatives', 1)}** this "
                   "session), so retraining will teach me to stop mistaking images "
                   "like it for trains.\n\nUpload another image whenever you're ready.")
        else:
            bot = "Please reply **yes** or **no** — was the last image really a train?"

    # ---- Stage: waiting for an image ------------------------------------
    else:  # stage == "ask_image"
        if files:
            fpath = files[0]
            n_trains, confs, label_lines = detect_trains(fpath)
            result_image = RESULT_IMG
            name = state.get("name", "there")

            if n_trains == 0:
                # STOP here — no train, so no OCR.
                bot = (f"🚫🚆 Sorry {name}, there's **no train** in this picture. "
                       "I stopped here (no text to scan). Please upload a different image.")
            else:
                # Train found -> read text, then ask for confirmation.
                texts = read_text(fpath)
                if n_trains == 1:
                    bot = f"I found **1 train** (confidence {confs[0]:.0%})."
                else:
                    cs = ", ".join(f"{c:.0%}" for c in confs)
                    bot = f"I found **{n_trains} trains** (confidences: {cs})."
                if texts:
                    joined = ", ".join(f"`{t}`" for t in texts)
                    bot += f"\n\nText I could read: {joined}"
                else:
                    bot += "\n\nI didn't find any readable text on it."

                # Stage the image so it survives until the user answers.
                staged = os.path.join(STAGING_DIR, f"stage_{int(time.time()*1000)}.jpg")
                shutil.copy2(fpath, staged)
                state["pending"] = {"image": staged, "labels": label_lines}
                state["stage"] = "confirm"
                bot += ("\n\n❓ **Was I right — is this actually a train?** "
                        "Reply **yes** or **no**.")
        elif text.lower() == "reset":
            state.update({"stage": "ask_name", "name": None, "pending": None})
            bot = "Okay, let's start over. What's your name?"
        elif text:
            bot = "Just send me an image using the 📎 clip. (Type **reset** to start over.)"
        else:
            bot = "Please upload an image using the 📎 clip. 🚆"

    history = history + [{"role": "assistant", "content": bot}]
    return {"text": "", "files": []}, history, state, result_image


def retrain_now(state):
    """Kick off background retraining on all confirmed samples."""
    if TRAIN_STATUS["busy"]:
        return "🟠 Already retraining — please wait for it to finish."
    new_count = _count_user_samples()
    if new_count == 0:
        return "ℹ️ No confirmed samples yet. Confirm a train with **yes** first."
    threading.Thread(target=_retrain_worker, args=(30,), daemon=True).start()
    return (f"🟠 Retraining started in the background on {new_count} confirmed "
            "image(s). The chat still works; I'll switch to the improved model "
            "when it's done. (This can take a while on CPU.)")


def check_status():
    """Return the latest training status message."""
    if TRAIN_STATUS["busy"]:
        return TRAIN_STATUS["msg"] or "🟠 Retraining in progress…"
    return TRAIN_STATUS["msg"] or "⚪ Idle."


def _count_user_samples():
    return len([f for f in os.listdir(TRAIN_IMG_DIR)
                if f.startswith("user_") or f.startswith("neg_")])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Train Detection Chatbot") as demo:
    gr.Markdown("# 🚆 Train Detection Chatbot\n"
                "I detect trains with your YOLO model, read text with PaddleOCR, "
                "and learn from images you confirm.")

    state = gr.State({"stage": "ask_name", "name": None, "pending": None, "new_samples": 0})

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=480,
                value=[{"role": "assistant",
                        "content": "Hello! 👋 I'm your train-detection assistant. "
                                   "What's your name?"}],
            )
            msg = gr.MultimodalTextbox(
                placeholder="Type here… (attach an image with the 📎 clip)",
                file_types=["image"],
                show_label=False,
            )
        with gr.Column(scale=2):
            result_view = gr.Image(label="Detection result", height=360)
            retrain_btn = gr.Button("🔁 Retrain now (learn from confirmed images)")
            status_md = gr.Markdown("⚪ Idle.")
            refresh_btn = gr.Button("🔄 Check training status", size="sm")

    msg.submit(chat_fn, [msg, chatbot, state], [msg, chatbot, state, result_view])
    retrain_btn.click(retrain_now, [state], [status_md])
    refresh_btn.click(check_status, None, [status_md])


if __name__ == "__main__":
    demo.launch(inbrowser=False, theme=gr.themes.Soft())
