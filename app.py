from flask import Flask, render_template, request, url_for
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import os
import gdown

# --- KRİTİK EKLEME 1: Render Sunucu Ayarı ---
import matplotlib
matplotlib.use('Agg') # Sunucuda grafik çizerken hata almamak için şart!
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# --- Klasör Ayarları ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
GRAPH_FOLDER = os.path.join(BASE_DIR, "static", "graphs")

for folder in [UPLOAD_FOLDER, GRAPH_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRAPH_FOLDER'] = GRAPH_FOLDER

# --- Model İndirme ---
MODEL_PATH = "pneumonia_model.keras"
FILE_ID = "11Bt0nfupPs4T6G4xxMoW6mD0eqg-bBgK"

if not os.path.exists(MODEL_PATH):
    print("Model indiriliyor...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# --- KERAS 3 -> 2 UYUMLULUK YAMASI ---
def fixed_from_config(cls, config):
    config.pop('batch_shape', None)
    config.pop('optional', None)
    config.pop('quantization_config', None)
    config.pop('registered_name', None)
    return cls(**config)

target_layers = [tf.keras.layers.InputLayer, tf.keras.layers.Dense, 
                 tf.keras.layers.Conv2D, tf.keras.layers.MaxPooling2D, 
                 tf.keras.layers.Flatten]

for layer in target_layers:
    layer.from_config = classmethod(fixed_from_config)

# --- Model Yükleme ---
# KRİTİK EKLEME 2: Modeli global olarak tanımlıyoruz
model = None 

print("Model yükleniyor...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model başarıyla yüklendi!")
except Exception as e:
    print(f"Yükleme hatası: {e}")

# Metrics yükleme
try:
    metrics = pickle.load(open("metrics.pkl", "rb"))
except:
    metrics = {"auc": 0.0} # Dosya yoksa hata vermemesi için

def prepare(img):
    img = img.resize((150,150))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Modelin fonksiyon içinde tanınması için global diyoruz
    global model 
    
    if model is None:
        return "Model yüklenemedi, lütfen logları kontrol edin.", 500

    file = request.files["file"]
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    img = Image.open(path).convert("RGB")
    img_prepared = prepare(img)
    
    # Tahmin
    pred_prob = model.predict(img_prepared)[0][0]
    result = "PNEUMONIA" if pred_prob > 0.5 else "NORMAL"

    # ROC Çizimi
    y_true = np.array([1])  
    y_scores = np.array([pred_prob])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc_live = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"Live ROC AUC = {roc_auc_live:.2f}")
    plt.plot([0,1],[0,1], linestyle='--')
    plt.title("Live ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    
    live_roc_path = os.path.join(app.config['GRAPH_FOLDER'], "live_roc.png")
    plt.savefig(live_roc_path)
    plt.close() # Belleği temizlemek için şart

    return render_template("result.html",
                           result=result,
                           metrics=metrics,
                           roc=metrics.get("auc", 0),
                           image=file.filename,
                           live_roc="live_roc.png"
                           )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
