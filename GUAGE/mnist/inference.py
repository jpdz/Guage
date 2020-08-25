import tensorflow as tf
from tensorflow import keras
import cv2

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#########
tflite_mnist_model = "mnist.tflite"

loaded_digit = cv2.imread(path_to_images)

loaded_digit = np.expand_dims(loaded_digit, axis=0)
loaded_digit = np.expand_dims(loaded_digit, axis=3)
loaded_digit.shape

interpreter = tf.lite.Interpreter(model_path=tflite_mnist_model)
interpreter.allocate_tensors()

loaded_digit = loaded_digit.astype('float32')
input_details = interpreter.get_input_details()
interpreter.set_tensor(input_details[0]['index'], loaded_digit)
interpreter.invoke()
output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Prediction results:", output_data)
print("Predicted value:", np.argmax(output_data))
