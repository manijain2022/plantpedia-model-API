import os
import gdown
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = FastAPI()

# --- Download model from GDrive if not already ---
def download_model_from_gdrive():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    model_filename = "E35_D5_LeakyRelu_0.0005_A94.h5"
    destination = os.path.join(model_dir, model_filename)

    if os.path.exists(destination):
        print("✅ Model file already exists.")
        return destination

    print("⬇️  Downloading model from Google Drive...")
    url = "https://drive.google.com/uc?id=1AYaUN4QAskXZaN8kjv7Y92WjSgrUsCNC"
    gdown.download(url, destination, quiet=False)
    print("✅ Download complete.")
    return destination

model_path = download_model_from_gdrive()
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU

model = load_model(model_path, compile=False, custom_objects={"LeakyReLU": LeakyReLU})

# model = load_model(model_path, compile=False, custom_objects={"LeakyReLU": LeakyReLU})

# --- Class Names ---
class_names = ['Aloevera', 'Amar poi', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha', 'Astma_weed', 
               'Avacado', 'Badipala', 'Balloon_Vine', 'Bamboo', 'Basale', 'Beans', 'Betel', 'Betel_Nut', 
               'Bhrami', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus', 'Chakte', 'Chilly', 
               'Citron lime (herelikai)', 'Coffee', 'Common rue', 'Coriender', 'Curry_Leaf', 'Doddapatre', 
               'Drumstick', 'Ekka', 'Eucalyptus', 'Ganike', 'Gasagase', 'Geranium', 'Ginger', 
               'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 
               'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemon_grass', 'Malabar_Nut', 'Mango', 'Marigold', 
               'Mint', 'Nagadali', 'Neem', 'Nelavembu', 'Nerale', 'Nooni', 'Onion', 'Padri', 
               'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper', 'Pomegranate', 'Pumpkin', 'Raddish', 
               'Raktachandini', 'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala', 'Tamarind', 'Taro', 
               'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric', 'Wood_sorel', 'camphor', 'kamakasturi', 'kepala'] # use same list as before (trimmed here for brevity)

# --- API Route ---
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]

    return JSONResponse({"prediction": predicted_class})