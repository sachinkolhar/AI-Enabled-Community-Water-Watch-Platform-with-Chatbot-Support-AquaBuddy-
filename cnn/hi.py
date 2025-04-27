import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# 1. Load the saved model locally (change the path if necessary)
model = tf.keras.models.load_model('D:/RVU/Sem_4/hackthon/IEEE RVU/waterflask/cnn/saved_model.h5')  # Adjust the path

# 2. Define class labels (based on training folders order)
class_labels = ['pollution', 'scarcity', 'misuse']

# 3. Function to predict a single image
def predict_image(img_path):
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # (1, 150, 150, 3)
        img_array = img_array / 255.0  # Rescale to [0, 1]

        # Predict
        prediction = model.predict(img_array)
        print(f"Raw prediction probabilities: {prediction}")
        predicted_class = class_labels[np.argmax(prediction)]

        if np.max(prediction) < 0.5:  # Confidence threshold
            print("⚠️ Low confidence in prediction.")
            
        print(f"🔍 Prediction: {predicted_class.capitalize()}")
        return predicted_class
    except Exception as e:
        print(f"Error: {e}")

# 4. Example usage:
predict_image("D:/RVU/Sem_4/hackthon/IEEE RVU/waterflask/cnn/download.jpg")  # Adjust the path to your test image
