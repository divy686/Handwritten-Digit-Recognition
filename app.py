from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
import os

# Flask setup
app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')  
CORS(app)

# Load trained CNN model
model = load_model('digit_cnn_model.h5')

# Store previous predictions
history = []

# Folder to save uploaded images (optional)
UPLOAD_FOLDER = "uploaded_images"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(img_data):
    """
    Converts base64-encoded image (from canvas or upload)
    into a 28x28 grayscale array ready for model prediction.
    """
    # If image is base64 (from canvas)
    if isinstance(img_data, str) and img_data.startswith('data:image'):
        img_bytes = base64.b64decode(img_data.split(',')[1])
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
    else:
        # If image uploaded as file (bytes)
        img = Image.open(io.BytesIO(img_data)).convert('L')
    
    # Resize to 28x28 and convert to numpy array
    img = img.resize((28, 28))
    img_array = np.array(img)

    # Auto-detect background and invert if needed
    if np.mean(img_array) > 127:  # Means white background
        img_array = 255 - img_array

    # Normalize
    img_array = img_array / 255.0

    # Optional: threshold cleanup (remove light noise)
    img_array[img_array < 0.2] = 0
    img_array[img_array > 0.8] = 1

    return img_array.reshape(1, 28, 28, 1)


# ---------------- HOME ROUTE ----------------
@app.route('/')
def home():
    return render_template('index.html')


# ---------------- PREDICT (Canvas or Upload) ----------------
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles both canvas-drawn image and uploaded image predictions.
    """
    if 'file' in request.files:  # Image upload
        file = request.files['file']
        img_data = file.read()
        img_array = preprocess_image(img_data)
    else:  # Image from canvas
        data = request.get_json()
        img_array = preprocess_image(data['image'])

    # Predict digit
    prediction = model.predict(img_array)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    # Confidence message
    message = "Prediction confident ✅" if confidence >= 0.7 else "Model unsure 🤔"

    # Save to history
    history.append({
        'digit': digit,
        'confidence': round(confidence, 2),
        'time': datetime.now().strftime('%H:%M:%S')
    })
    if len(history) > 10:  # Keep only last 10 entries
        history.pop(0)

    return jsonify({
        'digit': digit,
        'confidence': round(confidence, 2),
        'message': message,
        'history': history
    })


# ---------------- CLEAR HISTORY ----------------
@app.route('/clear_history', methods=['POST'])
def clear_history():
    """
    Clears prediction history.
    """
    history.clear()
    return jsonify({"message": "Prediction history cleared!"})


# ---------------- UPLOAD ENDPOINT (OPTIONAL) ----------------
@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Endpoint to handle direct image uploads from user (optional advanced feature).
    """
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return jsonify({"message": f"File '{file.filename}' uploaded successfully!"})


# ---------------- MAIN ----------------
if __name__ == '__main__':
    app.run(debug=True)
