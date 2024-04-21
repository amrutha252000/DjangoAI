from django.shortcuts import render
from django.http import JsonResponse
import base64
import io
import numpy as np
from PIL import Image
import json
import cv2
import os
# Create your views here.
from django.http import HttpResponse
import tensorflow as tf
from django.views.decorators.csrf import csrf_exempt


def index(request):
    return render(request, 'index.html')

@csrf_exempt
def get_prediction(request):
    print("hi")
    model = tf.keras.models.load_model('gesture_recog_vgg.keras')
    model.summary()
    if request.method == 'POST':
        print(os.getcwd())
        image_data = json.loads(request.body)['image_data']

        # Decode base64 data
        image_bytes = base64.b64decode(image_data.split(',')[1])  # Remove header if present
        image_stream = io.BytesIO(image_bytes)

        # Load image into NumPy array
        img = Image.open(image_stream)
        np_array = np.asarray(img)
        print(np_array.shape)
        # ... Your image processing with the np_array ...
        # if len(np_array.shape) == 3:  # Check if already grayscale
        #     np_array = cv2.cvtColor(np_array, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        np_array = np.array([cv2.resize(np_array, (224, 224), interpolation=cv2.INTER_AREA)])
        print(np_array.shape)
        resp = model.predict(np_array)[0].tolist()
        print(resp)
        return JsonResponse({'resp': resp}) 
    else:
        return JsonResponse({'error': 'Invalid request method'})