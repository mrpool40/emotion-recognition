# Colab inline dashboard: paste image URL -> shows annotated image + predictions
# Run this whole cell in a Colab notebook

# 0. Install required packages (only first time)
!pip install -q huggingface-hub tensorflow opencv-python-headless ipywidgets matplotlib requests

# 1. Imports & model download/load
import io, os, requests, threading
import numpy as np
import cv2
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
import ipywidgets as widgets
from IPython.display import display, clear_output

# 2. Globals / constants
HF_REPO = "shivamprasad1001/Emo0.1"
HF_FILE = "Emo0.1.h5"
EMOTIONS = ['angry','disgusted','fearful','happy','neutral','sad','surprised']
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# 3. Load model (cached)
print("Downloading/Loading model (cached)... this may take a few seconds on first run.")
model_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
model = load_model(model_path)
print("Model loaded:", model_path)

# Helper: infer model input H,W,C
def infer_model_io(m):
    try:
        inp = m.input_shape
    except Exception:
        inp = tuple([int(d) if d is not None else None for d in m.inputs[0].shape])
    if len(inp) == 4:
        _, H, W, C = inp
    elif len(inp) == 3:
        a,b,c = inp
        if a in (1,3):
            C,H,W = int(a),int(b),int(c)
        elif c in (1,3):
            H,W,C = int(a),int(b),int(c)
        else:
            H,W,C = int(a),int(b),int(c)
    else:
        H,W,C = 48,48,1
    H = int(H) if H is not None else 48
    W = int(W) if W is not None else 48
    C = int(C) if C is not None else 1
    return H,W,C

H_model, W_model, C_model = infer_model_io(model)
print(f"Model expects H={H_model}, W={W_model}, C={C_model}")

# Preprocessing that adapts to model's channel count
def prepare_face(face_bgr, target_h, target_w, target_c):
    """face_bgr: BGR image (numpy)"""
    if face_bgr is None:
        raise ValueError("face is None")
    if len(face_bgr.shape) == 2:
        bgr = cv2.cvtColor(face_bgr, cv2.COLOR_GRAY2BGR)
    else:
        bgr = face_bgr
    img = cv2.resize(bgr, (target_w, target_h))  # cv2 uses (w,h)
    if target_c == 1:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")/255.0
        arr = np.expand_dims(g, axis=(0,-1))  # 1,H,W,1
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")/255.0
        arr = np.expand_dims(rgb, axis=0)     # 1,H,W,3
    return arr

def predict_face(face_bgr):
    x = prepare_face(face_bgr, H_model, W_model, C_model)
    preds = model.predict(x)
    probs = preds[0]
    idx = int(np.argmax(probs))
    return EMOTIONS[idx] if idx < len(EMOTIONS) else str(idx), float(probs[idx]), probs

# Face detector
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# 4. Build widgets: URL input, button, file uploader, results output
url_box = widgets.Text(value='', placeholder='Paste an image URL (http/https) here', description='Image URL:', layout=widgets.Layout(width='80%'))
upload = widgets.FileUpload(accept='image/*', multiple=False)
run_btn = widgets.Button(description='Detect & Predict', button_style='primary')
out = widgets.Output(layout=widgets.Layout(border='1px solid gray'))

# Helper to load image from URL or uploaded bytes
def load_image_from_url(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    arr = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def load_image_from_upload(u):
    if not u:
        return None
    # u is dict-like: pick first file
    b = list(u.values())[0]['content'] if isinstance(u, dict) else next(iter(u.values()))  # compatibility
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

# Action when button clicked
def on_run_clicked(b):
    with out:
        clear_output(wait=True)
        print("Processing...")
        # get image
        img = None
        if url_box.value.strip():
            try:
                img = load_image_from_url(url_box.value.strip())
            except Exception as e:
                print("Failed to download image from URL:", e)
                img = None
        if img is None and upload.value:
            # upload.value is a dict mapping filename->metadata
            try:
                # Colab's FileUpload returns .value as dict in Jupyter; in Colab it is a list-like; handle both
                if isinstance(upload.value, dict):
                    img = load_image_from_upload(upload.value)
                else:
                    # newer ipywidgets returns a list of dicts
                    content = list(upload.value)[0]['content'] if isinstance(list(upload.value)[0], dict) else list(upload.value)[0]
                    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                print("Failed to read uploaded file:", e)
                img = None

        if img is None:
            print("No valid image loaded. Paste a valid URL or upload an image file.")
            return

        # Detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        annotated = img.copy()

        results = []
        if len(faces) == 0:
            # fallback: predict on full image
            label, prob, probs = predict_face(img)
            print(f"No faces found — predicted overall image: {label} ({prob:.3f})")
            # show image
            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis('off')
            plt.show()
            return

        # For each face, crop and predict
        for i,(x,y,w,h) in enumerate(faces):
            face = img[y:y+h, x:x+w]
            label, prob, probs = predict_face(face)
            results.append((i+1, label, prob, (x,y,w,h)))
            # draw bounding box + label
            cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(annotated, f"{label} {prob:.2f}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Display annotated image
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        plt.show()

        # Print table of detected faces
        from IPython.display import display
        import pandas as pd
        df = pd.DataFrame([{'face': r[0], 'label': r[1], 'confidence': r[2]} for r in results])
        display(df)

# Attach handler
run_btn.on_click(on_run_clicked)

# 5. Layout
title = widgets.HTML("<h3>Colab Inline Emotion Detection Dashboard</h3><p>Paste image URL or upload image, then click <b>Detect & Predict</b>.</p>")
controls = widgets.HBox([url_box, run_btn])
uploader_row = widgets.HBox([widgets.Label("Or upload:"), upload])
display(title, controls, uploader_row, out)
