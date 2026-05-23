import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the model
model = tf.keras.models.load_model('cracked_phone_detector_convnext.keras')

def predict_phone(image_path):
    # 2. Load and Preprocess the image
    # We must resize it to the exact same size used in training (224, 224)
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    
    # 3. Add a 'batch dimension' 
    # The model expects [batch_size, height, width, channels]
    # We have 1 image, so shape becomes
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Run Prediction
    predictions = model.predict(img_array)
    score = predictions # The sigmoid output (value between 0 and 1)

    # 5. Interpret the class mapping
    # Note: image_dataset_from_directory sorts folders alphabetically
    # 'cracked' comes before 'not_cracked', so:
    # index 0 (low score) = Cracked | index 1 (high score) = Not Cracked
    # Unless you swapped them, Sigmoid < 0.5 usually means the first alphabetical class.
    
    if score < 0.5:
        label = "CRACKED"
        confidence = (1 - score) * 100
    else:
        label = "NOT CRACKED"
        confidence = score * 100

    # 6. Show the result
    plt.imshow(img)
    plt.title(f"Result: {label} ({confidence:.2f}% confidence)")
    plt.axis('off')
    plt.show()
    
    print(f"The model is {confidence:.2f}% confident that the phone is {label}")

# --- RUN IT ---
predict_phone('test_image.jpg') # Replace with your image file name