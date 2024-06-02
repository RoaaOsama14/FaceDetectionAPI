from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('E:\\Graduation Project\\FaceRecognition_API\\my_modelonall.keras')

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def index():
    return 'Welcome to the Face Recognition API!'
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        img_array = prepare_image(file_path)
        prediction = model.predict(img_array)

        os.remove(file_path)  # Clean up the uploaded file

        if prediction[0] > 0.5:
            predicted_label = "Not a face"
        else:
            predicted_label = "Face"

        return jsonify({'prediction': predicted_label})

    return jsonify({'error': 'Invalid request'}), 400

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
