from flask import Flask, render_template, request, url_for
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


UPLOAD_FOLDER = os.path.join("static", "uploads")
GRAPH_FOLDER = os.path.join("static", "graphs")

for folder in [UPLOAD_FOLDER, GRAPH_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRAPH_FOLDER'] = GRAPH_FOLDER


import gdown

MODEL_PATH = "pneumonia_model.keras"
FILE_ID = "11Bt0nfupPs4T6G4xxMoW6mD0eqg-bBgK"

if not os.path.exists(MODEL_PATH):
    print("Model indiriliyor...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

print("Model yükleniyor...")

from tensorflow.keras.layers import Dense, InputLayer

# Keras 3 ile kaydedilen modellerdeki bilinmeyen parametreleri temizleyen sınıflar
class FixedDense(Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)

class FixedInputLayer(InputLayer):
    def __init__(self, **kwargs):
        kwargs.pop('batch_shape', None)
        kwargs.pop('optional', None)
        super().__init__(**kwargs)

# Keras'ın bu hatalı sınıflar yerine bizim "temizlenmiş" sınıflarımızı kullanmasını sağlıyoruz
tf.keras.utils.get_custom_objects()['Dense'] = FixedDense
tf.keras.utils.get_custom_objects()['InputLayer'] = FixedInputLayer

print("Model yükleniyor...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)



metrics = pickle.load(open("metrics.pkl", "rb"))


def prepare(img):
    img = img.resize((150,150))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

# Ana Sayfa Yönlendirme
@app.route("/")
def home():
    return render_template("index.html")

# Proje Hakkında Sayfası Yönlendirme
@app.route("/about")
def about():
    return render_template("about.html")

# Sonuç Yönlendirme
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    
    img = Image.open(path).convert("RGB")
    img_prepared = prepare(img)
    pred_prob = model.predict(img_prepared)[0][0]
    result = "PNEUMONIA" if pred_prob > 0.5 else "NORMAL"

    
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
    plt.close()

    return render_template("result.html",
                           result=result,
                           metrics=metrics,
                           roc=metrics["auc"],
                           image=file.filename,
                           live_roc="live_roc.png"
                           )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
