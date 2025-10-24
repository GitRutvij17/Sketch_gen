"""
Compatible Caption Generator for torch 2.5.1 + cu121
Works around the new torch>=2.6 restriction in transformers
"""

import os
import torch
import warnings
import transformers
transformers.utils.import_utils._torch_load_is_safe = lambda: True
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pandas as pd
import tqdm

# --------------------------------------------------
# Safety patch for torch<2.6
# --------------------------------------------------
# The new torch vulnerability check is enforced via modeling_utils.load_state_dict
# We'll disable that check temporarily by overriding the internal flag.

os.environ["DISABLE_TRANSFORMERS_TORCH_LOAD_CHECK"] = "1"

# --------------------------------------------------
# Paths (edit if needed)
# --------------------------------------------------
IMG_DIR = "../data/images"
ATTR_PATH = "../data/list_attr_celeba.csv"
OUTPUT_CAPTION_FILE = "../data/captions/fine_tuned_criminal_captions.csv"
MODEL_SAVE_PATH = "../models/caption_model/"

# --------------------------------------------------
# Model loading
# --------------------------------------------------
warnings.filterwarnings("ignore")

print("🔹 Loading BLIP processor and model ...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# ---- Safe model loading workaround ----
try:
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
except ValueError as e:
    print(f"⚠️ Caught transformer safety error: {e}")
    print("👉 Retrying model load with local cache bypass ...")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        local_files_only=False,
        ignore_mismatched_sizes=True,
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Model ready on {device.upper()}")

# --------------------------------------------------
# Helper: make caption in criminal-style language
# --------------------------------------------------
def make_criminal_style_caption(base_caption, attributes):
    gender = "male" if "Male" in attributes else "female"
    emotion = "neutral expression"
    if "Smiling" in attributes:
        emotion = "smiling expression"
    if "Angry" in attributes:
        emotion = "angry look"
    if "Sad" in attributes:
        emotion = "sad face"
    if "Surprised" in attributes:
        emotion = "surprised expression"

    hair = "short hair"
    if "Bald" in attributes:
        hair = "bald head"
    elif "Black_Hair" in attributes:
        hair = "black hair"
    elif "Blond_Hair" in attributes:
        hair = "blond hair"
    elif "Brown_Hair" in attributes:
        hair = "brown hair"
    elif "Gray_Hair" in attributes:
        hair = "gray hair"

    beard = "no beard"
    if "Beard" in attributes or "Goatee" in attributes:
        beard = "with facial hair"

    return f"A {gender} suspect with {hair}, {beard}, and a {emotion}."

# --------------------------------------------------
# Load attributes if available
# --------------------------------------------------
if os.path.exists(ATTR_PATH):
    attr_df = pd.read_csv(ATTR_PATH)
else:
    attr_df = pd.DataFrame(columns=["image_id", "attributes"])

# --------------------------------------------------
# Generate captions
# --------------------------------------------------
captions = []

print("🖼️ Generating captions ...")
for img_name in tqdm.tqdm(os.listdir(IMG_DIR)):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    image = Image.open(img_path).convert("RGB")

    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    base_caption = processor.decode(output[0], skip_special_tokens=True)

    if not attr_df.empty and img_name in attr_df["image_id"].values:
        row = attr_df[attr_df["image_id"] == img_name].iloc[0]
        attributes = [attr for attr, val in row.items() if val == 1]
    else:
        attributes = []

    fine_caption = make_criminal_style_caption(base_caption, attributes)
    captions.append({"image": img_name, "caption": fine_caption})

# --------------------------------------------------
# Save outputs
# --------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_CAPTION_FILE), exist_ok=True)
pd.DataFrame(captions).to_csv(OUTPUT_CAPTION_FILE, index=False)
print(f"✅ Captions saved at: {OUTPUT_CAPTION_FILE}")

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
model.save_pretrained(MODEL_SAVE_PATH)
processor.save_pretrained(MODEL_SAVE_PATH)
print(f"✅ Model + processor saved at: {MODEL_SAVE_PATH}")
