import sys
import os
import cv2
import numpy as np
from PIL import Image
from tkinter import Tk, filedialog

import tensorflow as tf

# ---------- Config ----------
MODEL_PATH = "cattle_breed_classifier.h5"
TRAIN_DIR = "IndianCattleBuffaloeBreeds-Dataset/breeds/train" 
OUT_FILE = "annotated_result.jpg"
AUG_ROT_ANGLES = [0, -5, 5]   
DO_MOBILENET_PREPROCESS = True  
# ----------------------------

def load_model_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = tf.keras.models.load_model(path)
    return model

def get_class_names(train_dir):
    if os.path.isdir(train_dir):
        names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        if len(names) > 0:
            return names
    txt = "class_names.txt"
    if os.path.exists(txt):
        with open(txt, "r") as f:
            return [l.strip() for l in f.readlines() if l.strip()]
    raise FileNotFoundError("Could not find class names. Make sure TRAIN_DIR is correct or provide class_names.txt")

def choose_image_path():
    if len(sys.argv) > 1:
        p = sys.argv[1]
        if os.path.exists(p):
            return p
        else:
            print("File not found:", p)
            sys.exit(1)
    # else open file dialog
    Tk().withdraw()
    file_path = filedialog.askopenfilename(title="Select image",
                                           filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp")])
    if not file_path:
        print("No image selected. Exiting.")
        sys.exit(0)
    return file_path

def load_image_cv(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("cv2 failed to read the image (maybe corrupted).")
    return img

def rotate_image(img, angle):
    if angle == 0:
        return img.copy()
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def prepare_array(img_rgb, target_size, method="rescale"):
    # img_rgb is HxWx3 RGB uint8
    img_resized = cv2.resize(img_rgb, (target_size[1], target_size[0]))  # (w,h) vs (h,w)
    arr = img_resized.astype("float32")
    if method == "rescale":
        arr /= 255.0
    elif method == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        arr = preprocess_input(arr)
    else:
        raise ValueError("Unknown preprocess method")
    return np.expand_dims(arr, axis=0)

def predict_with_augmentations(model, orig_bgr, target_size, class_count, preprocess_method):
    # orig_bgr: whole-image in BGR (cv2)
    preds_accum = np.zeros((class_count,), dtype=np.float32)
    n = 0
    for angle in AUG_ROT_ANGLES:
        rot = rotate_image(orig_bgr, angle)
        for flip in (False, True):
            img_variant = cv2.flip(rot, 1) if flip else rot
            img_rgb = cv2.cvtColor(img_variant, cv2.COLOR_BGR2RGB)
            arr = prepare_array(img_rgb, target_size, method=preprocess_method)
            try:
                p = model.predict(arr, verbose=0)[0]
            except Exception as e:
                raise RuntimeError(f"Model predict failed: {e}")
            preds_accum += p
            n += 1
    if n == 0:
        raise RuntimeError("No augmentations generated.")
    return preds_accum / n

def main():
    print("Loading model...")
    model = load_model_safe(MODEL_PATH)
    try:
        inp_shape = model.input_shape  # typically (None, H, W, 3)
        if isinstance(inp_shape, list):
            inp_shape = inp_shape[0]
        target_h = int(inp_shape[1]) if inp_shape[1] is not None else 224
        target_w = int(inp_shape[2]) if inp_shape[2] is not None else 224
    except Exception:
        target_h, target_w = 224, 224
    target_size = (target_h, target_w)
    print(f"Model input size -> {target_size}")

    # class names
    try:
        class_names = get_class_names(TRAIN_DIR)
    except Exception as e:
        print("Warning: couldn't load class names from train folder:", e)
        class_names = None

    if class_names is None:
        print("Could not get class names automatically. I will infer number of classes from model output.")
        dummy_out = model.predict(np.zeros((1, target_h, target_w, 3), dtype=np.float32))
        num_classes = int(dummy_out.shape[1])
        class_names = [f"class_{i}" for i in range(num_classes)]
    num_classes = len(class_names)

    image_path = choose_image_path()
    orig = load_image_cv(image_path)
    h, w = orig.shape[:2]
    print(f"Image loaded: {image_path}  (W x H = {w} x {h})")

    preds_rescale = predict_with_augmentations(model, orig, target_size, num_classes, preprocess_method="rescale")
    if DO_MOBILENET_PREPROCESS:
        try:
            preds_mobilenet = predict_with_augmentations(model, orig, target_size, num_classes, preprocess_method="mobilenet")
        except Exception as e:
            print("Mobilenet preprocess attempt failed:", e)
            preds_mobilenet = None
    else:
        preds_mobilenet = None


    top_rescale = float(np.max(preds_rescale))
    top_mob = float(np.max(preds_mobilenet)) if preds_mobilenet is not None else -1.0

    if preds_mobilenet is not None and top_mob > top_rescale:
        preds_final = preds_mobilenet
        chosen_method = "mobilenet_preprocess"
        top_conf = top_mob
    else:
        preds_final = preds_rescale
        chosen_method = "rescale_1_by_255"
        top_conf = top_rescale

    idx = int(np.argmax(preds_final))
    predicted_label = class_names[idx]
    confidence = float(preds_final[idx])


    annotated = orig.copy()
    border_color = (0, 200, 0)
    thickness = max(2, min(w, h) // 200)
    cv2.rectangle(annotated, (1,1), (w-2, h-2), border_color, thickness)

    # build text box
    text = f"{predicted_label} ({confidence*100:.1f}%)"
    small_text = f"method:{chosen_method}"
    # put a filled rect behind text for readability
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, min(w, h) / 800.0)
    t_size = cv2.getTextSize(text, font, scale, 2)[0]
    pad = 8
    x0, y0 = 10, 10 + t_size[1]
    cv2.rectangle(annotated, (5,5), (5 + t_size[0] + pad*2, 5 + t_size[1] + pad*2), (0,0,0), -1)
    cv2.putText(annotated, text, (10 + pad, 10 + t_size[1] + pad//2), font, scale, (255,255,255), 2, cv2.LINE_AA)

    # show and save
    try:
        cv2.imwrite(OUT_FILE, annotated)
        print(f"Annotated image saved to: {OUT_FILE}")
    except Exception as e:
        print("Warning: could not save annotated image:", e)

    # Try to display (may fail on headless systems)
    try:
        cv2.imshow("Prediction (whole image)", annotated)
        print("Press any key in the image window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("Cannot open image window (headless?). Saved annotated image instead.")

    # Print detailed results
    print("\n=== Prediction result ===")
    print("Label:", predicted_label)
    print(f"Confidence: {confidence*100:.2f}%   (chosen preprocess: {chosen_method})")
    print("\nTop-5 scores:")
    top5_idx = np.argsort(preds_final)[::-1][:5]
    for i in top5_idx:
        print(f"  {class_names[i]:30s} {preds_final[i]*100:6.2f}%")

if __name__ == "__main__":
    main()
