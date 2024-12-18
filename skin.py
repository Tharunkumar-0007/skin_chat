import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained skin disease model
model = load_model('image_classifier_model.h5')

# Load class labels from diseases.json
with open('diseases.json', 'r') as f:
    class_labels = json.load(f)

def preprocess_image(image_file):
    """Preprocess the image for prediction."""
    img = Image.open(image_file).resize((150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def validate_skin_image_by_color(image_file):
    """Validate if the image contains skin-related content using color detection."""
    # Convert the uploaded file into a NumPy array
    image = np.array(Image.open(image_file).convert('RGB'))

    # Convert the image to YCrCb color space
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    # Define skin color range in YCrCb
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)

    # Mask the image for skin color
    skin_mask = cv2.inRange(image_ycrcb, lower_skin, upper_skin)

    # Calculate the percentage of skin pixels
    skin_percentage = np.sum(skin_mask > 0) / (skin_mask.size)

    # Threshold for skin detection
    return skin_percentage > 0.2  # At least 20% of the image should be skin

def predict_skin_disease(image_file):
    """Predict the skin disease from an image."""
    if not validate_skin_image_by_color(image_file):
        return {'message': 'Give a valid skin image'}

    # Preprocess and predict skin disease
    img_array = preprocess_image(image_file)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_probability = np.max(predictions)

    # Determine the predicted label
    threshold = 0.6
    if predicted_probability < threshold:
        predicted_label = 'Healthy Skin'
    else:
        predicted_label = class_labels.get(str(predicted_class), 'Not a valid image')

    return {'predicted_class': predicted_label}
