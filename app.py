import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import gdown

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, "static", "uploads")
app.config['GRAPH_FOLDER'] = os.path.join(BASE_DIR, "static", "graphs")

for folder in [app.config['UPLOAD_FOLDER'], app.config['GRAPH_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

MODEL_PATH = "pneumonia_model.keras"
FILE_ID = "11Bt0nfupPs4T6G4xxMoW6mD0eqg-bBgK"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

model = None

# MODEL YÜKLEME - EN SADE HALİ
print("Model yükleniyor...")
try:
    # Keras 3 dosyasını doğrudan, hiçbir yama olmadan yüklemeyi dene
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model başarıyla yüklendi!")
except Exception as e:
    print(f"Yükleme hatası: {e}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        return "Model yüklenemedi. Lütfen sistem yöneticisine danışın.", 500
    
    file = request.files["file"]
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    
    img = Image.open(path).convert("RGB").resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)
    result = "PNEUMONIA" if preds[0][0] > 0.5 else "NORMAL"
    
    # Boş bir grafik oluştur (hata vermemesi için)
    plt.figure()
    plt.title(f"Sonuç: {result}")
    plt.savefig(os.path.join(app.config['GRAPH_FOLDER'], "live_roc.png"))
    plt.close()
    
    return render_template("result.html", result=result, image=file.filename, live_roc="live_roc.png")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
