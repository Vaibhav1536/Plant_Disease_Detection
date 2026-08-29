import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
from huggingface_hub import hf_hub_download


# -----------------------------
# Load CSV files
# -----------------------------
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Download model from Hugging Face
# -----------------------------
MODEL_PATH = hf_hub_download(
    repo_id="vaibhav1536/plant_disease_detection_vaibha1536",
    filename="plant_disease_model_1_latest.pt"
)

# -----------------------------
# Load trained model
# -----------------------------
model = CNN.CNN(39)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=torch.device('cpu')
    )
)

model.eval()

# -----------------------------
# Prediction function
# -----------------------------
def prediction(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))

    with torch.no_grad():
        output = model(input_data)

    output = output.numpy()
    index = np.argmax(output)

    return index


# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)


@app.route('/')
def home_page():
    return render_template('home.html')


@app.route('/contact')
def contact():
    return render_template('contact-us.html')


@app.route('/index')
def ai_engine_page():
    return render_template('index.html')


@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')


@app.route('/submit', methods=['GET', 'POST'])
def submit():

    if request.method == 'POST':

        image = request.files['image']

        filename = image.filename

        upload_folder = 'static/uploads'
        os.makedirs(upload_folder, exist_ok=True)

        file_path = os.path.join(upload_folder, filename)

        image.save(file_path)

        print("Uploaded image:", file_path)

        pred = prediction(file_path)

        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]

        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent,
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image_url,
            buy_link=supplement_buy_link
        )


@app.route('/market', methods=['GET', 'POST'])
def market():

    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )


if __name__ == '__main__':
    app.run(debug=True)