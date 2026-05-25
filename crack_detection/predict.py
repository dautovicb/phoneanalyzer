import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('model/crack_detector.keras')

def predict_phone(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = predictions.item() # The sigmoid output (value between 0 and 1)
    
    if score < 0.5:
        label = "CRACKED"
        confidence = (1 - score) * 100
    else:
        label = "NOT CRACKED"
        confidence = score * 100
    
    print(f"The model is {confidence:.2f}% confident that the phone is {label}")


predict_phone('test.png')