from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf

# Define a flask app
app = Flask(__name__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
# Model saved with Keras model.save()
MODEL_PATH = 'D:\Downloads\Deep-learning-project\Plant_disease_detection\\final_modelcnn.h5'
classes=[
    'Apple__black_rot',
 'Apple__healthy',
 'Apple__rust',
 'Apple__scab',
  'Cherry__healthy',
 'Cherry__powdery_mildew',
 'Corn__common_rust',
 'Corn__gray_leaf_spot',
 'Corn__healthy',
 'Corn__northern_leaf_blight',
 'Grape__black_measles',
 'Grape__black_rot',
 'Grape__healthy',
 'Grape__leaf_blight_(isariopsis_leaf_spot)',
 'Mango__diseased',
 'Mango__healthy',

 'Peach__bacterial_spot',
 'Peach__healthy',
    'Pepper_bell__bacterial_spot',
  'Pepper_bell__healthy',
   'Pomegranate__diseased',
 'Pomegranate__healthy',
  'Potato__early_blight',
  'Potato__healthy',
 'Potato__late_blight',

 'Rice__brown_spot',
 'Rice__healthy',
 'Rice__hispa',
 'Rice__neck_blast',
  'Soybean__bacterial_blight',
  'Soybean__caterpillar',
  'Soybean__diabrotica_speciosa',
  'Soybean__healthy',
  'Soybean__mosaic_virus',
  'Soybean__powdery_mildew',
  'Soybean__rust',
 'Strawberry___leaf_scorch',
 'Strawberry__healthy',
 'Sugarcane__bacterial_blight',
 'Sugarcane__healthy',
 'Sugarcane__red_rot',
 'Sugarcane__red_stripe',
 'Sugarcane__rust',
  'Tea__algal_leaf',
  'Tea__bird_eye_spot',
  'Tea__brown_blight',  'Tea__healthy',
  'Tea__red_leaf_spot',
 'Tomato__bacterial_spot',
 'Tomato__early_blight',
 'Tomato__healthy',
 'Tomato__late_blight',
 'Tomato__leaf_mold',
 'Tomato__mosaic_virus',
 'Tomato__septoria_leaf_spot',
 'Tomato__spider_mites_(two_spotted_spider_mite)',
 'Tomato__yellow_leaf_curl_virus',
 'Wheat__brown_rust',
 'Wheat__healthy',
 'Wheat__septoria',
 'Wheat__yellow_rust'
]
# Load your trained model
model = load_model(MODEL_PATH)       # Necessary
# print('Model loaded. Start serving...')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet.preprocess_input
    )

    # Load the random image using the generator's flow_from_directory method
    # Use a DataFrame with just the path to the random image
    import pandas as pd
    randomimage_df = pd.DataFrame({'filename': [img_path]})

    random_image_generator = test_generator.flow_from_dataframe(
        dataframe=randomimage_df,
        x_col='filename',
        target_size=(224, 224),
        batch_size=1,  # Set batch size to 1 since we're dealing with a single image
        class_mode=None,  # No class labels are needed for a single image
        shuffle=False
    )

    # Generate the preprocessed image batch
    x = random_image_generator.next()

    # Get the predicted class index
    preds = model.predict(x)
    predicted_class_index = tf.argmax(preds, axis=1)[0]

    # Get the predicted class name
    predicted_class_name = classes[predicted_class_index]

    return predicted_class_name

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        prediction = model_predict(file_path, model)
        return prediction

    return None


if __name__ == '__main__':
    app.run(debug=False,threaded=False)

