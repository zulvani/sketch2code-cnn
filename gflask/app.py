from flask import Flask, jsonify, request
import cv2
import numpy as np
import pandas as pd
from flask_cors import CORS
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)  # Mengaktifkan CORS untuk semua endpoint


@app.route('/api/demo', methods=['GET'])
def demo():
    citra = cv2.imread('static/ui.png', cv2.IMREAD_GRAYSCALE)
    hasil = binarization(citra)
    contours, hierarchy = cv2.findContours(hasil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxs = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # top_left = (x, y)
        # bottom_right = (x + w, y + h)

        bounding_boxs.append([x,y, x+w, y+h])

    return jsonify(bounding_boxs)

@app.route('/api/classify', methods=['GET'])
def classify():
    model = load_model("cnn_model.keras")

    object = request.args.get('object') 

    # Load the image
    image_path = "static/" + object
    image = Image.open(image_path).convert("RGB")

    # Resize the image to match the model's input size (224x224)
    image = image.resize((224, 224))

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Normalize pixel values to the range [0, 1]
    image_array = image_array / 255.0

    # Add a batch dimension (model expects input shape [batch_size, height, width, channels])
    image_array = np.expand_dims(image_array, axis=0)

    # Predict the class probabilities
    predictions = model.predict(image_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])
    class_names = ['button', 'input-box', 'label']
    predicted_class_name = class_names[predicted_class_index]

    # print(f"Predicted class: {predicted_class_name}")

    return jsonify({'objectClass': predicted_class_name})

def binarization(citra):
    row = citra.shape[0]
    column = citra.shape[1]
    hasil = np.zeros((row, column), np.uint8)

    # Define a kernel for dilation
    kernel = np.ones((1, 1), np.uint8)  # 5x5 square kernel
    citra = cv2.dilate(citra, kernel, iterations=2)

    for i in range(row):
        for j in range(column):
            if citra[i, j] >= 254:
                hasil[i, j] = 0
            else:
                hasil[i, j] = 255    
    return hasil 

if __name__ == '__main__':
    app.run(debug=True)