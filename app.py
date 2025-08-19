from flask import Flask, request, jsonify, render_template
from PIL import Image
import cv2
import joblib
import os
import base64

app = Flask(__name__, static_folder='static')

# Load the trained K-NN model
model = joblib.load('model.pkl')

def preprocess_image(image_path, target_size=(100, 100)):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, target_size)
        gray_image = cv2.GaussianBlur(image, (5, 5), 0)
        return gray_image
    except Exception as e:
        return None

def identify_leopard(image_path):
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is not None:
        features = preprocessed_image.flatten()
        prediction = model.predict([features])
        return "Leopard detected!" if prediction == 1 else "Not a leopard.", preprocessed_image
    else:
        return "Error processing image.", None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/identify', methods=['POST'])
def identify():
    try:
        # Get the image file from the POST request
        image_file = request.files.get('image')

        if image_file:
            # Save the image to a temporary file
            temp_image_path = 'temp_image.jpg'
            image_file.save(temp_image_path)

            # Perform identification
            result, preprocessed_image = identify_leopard(temp_image_path)

            # Delete the temporary image file
            Image.open(temp_image_path).close()  # Close the image to release the file lock
            os.remove(temp_image_path)

            # Encode the image in base64
            _, image_buffer = cv2.imencode('.jpg', preprocessed_image)
            base64_image = base64.b64encode(image_buffer).decode('utf-8')

            return jsonify({'result': result, 'image': base64_image})

        return jsonify({'error': 'Image not found'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
