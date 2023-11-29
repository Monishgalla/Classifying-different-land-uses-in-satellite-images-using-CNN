from flask import Flask, render_template, request, jsonify
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image 
import os

app = Flask(__name__)


model = load_model('model.h5')  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        img_file = request.files['image']
        

        # Read and preprocess the image
        img = image.load_img(img_file, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the pixel values to be between 0 and 1

        # Open and preprocess the image using PIL
        img = Image.open(img_file)
        img = img.resize((64, 64))  # Resize the image to the desired size
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the pixel values to be between 0 and 1


        # Make prediction
        predictions = model.predict(img_array)

        # Get the predicted class
        predicted_class = np.argmax(predictions)

        # Get class names from your dataset
        class_names = [
    
    'agricultural1', 'airplane1', 'baseballdiamond1', 'beach1', 'buildings1',
    'chaparral1', 'denseresidential1', 'forest1', 'freeway1', 'golfcourse1',
    'harbor1', 'intersection1', 'mediumresidential1', 'mobilehomepark1',
    'overpass1', 'parkinglot1', 'river1', 'runway1', 'sparseresidential1',
    'storagetanks1', 'tenniscourt1'
]
  # Update with your actual class names

        
        result = {'class': class_names[predicted_class], 'confidence': float(predictions[0][predicted_class])}

        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':

    app.run(debug=True)
    

    # Use the PORT environment variable provided by Heroku or default to 5000
    port = int(os.environ.get("PORT", 8000))
    
    if __name__ == '__main__':
    app.run(port=port)
    


