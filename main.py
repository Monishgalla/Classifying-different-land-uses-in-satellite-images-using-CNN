from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the trained ResNet50 model
model = load_model('model.sav')

# Define class names
class_names = ["agricultural",
    "airplane",
    "baseball diamond",
    "beach",
    "buildings",
    "chaparral",
    "dense residential",
    "forest",
    "freeway",
    "golf course",
    "harbor",
    "intersection",
    "medium residential",
    "mobile home park",
    "overpass",
    "parking lot",
    "river",
    "runway",
    "sparse residential",
    "storage tanks",
    "tennis court"]  

def preprocess_image(image_path):
    img = Image.open(image_path).resize((64, 64))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        contents = await file.read()
        image_path = 'temp_image.png'
        with open(image_path, 'wb') as f:
            f.write(contents)

        # Preprocess the image
        img_array = preprocess_image(image_path)

        # Make predictions
        predictions = model.predict(img_array)

        # Get the predicted class
        predicted_class = np.argmax(predictions)

        # Get the class name
        class_name = class_names[predicted_class]

        return JSONResponse(content={"class_name": class_name, "probabilities": predictions.tolist()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
