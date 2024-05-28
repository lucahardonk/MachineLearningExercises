import os
import cv2
import numpy as np
import matplotlib as plot
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

myModel = 'my_handwritten_model' # saved in this directory

model = tf.keras.models.load_model(myModel)

mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
x_test  = tf.keras.utils.normalize(x_test, axis=1) # taking only the testing sample

# model evaluation using the reserved testing data
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy}")
print(f"Test loss: {loss}")

# recovering and predicting the images written by me

current_image_number = 0 
max_images = 9
image_directory = 'my_images'

while os.path.isfile(f"{image_directory}/image_{current_image_number}.png"):
    try:
        # Reads the image from the specified path and converts it to grayscale, I need a numpy array otherwise I get a shape error
        img = cv2.imread(f"{image_directory}/image_{current_image_number}.png")[:,:,0] # 0 is implicit conversion to grayscale
        # Inverts the colors of the image
        img = np.invert(np.array([img]))
        # Executes prediction using the model
        prediction = model.predict(img)
        # Prints the obtained prediction
        print(f"My humble prediction is: {np.argmax(prediction)}")
        # Displays the image in a window titled "Image"
        cv2.imshow('Image', img[0])
        # Waits for 2000 milliseconds (2 seconds) for a key press
        cv2.waitKey(2000)
        # Closes all OpenCV windows
        cv2.destroyAllWindows()
    except:
        print("Error")
    finally:
        current_image_number += 1 

