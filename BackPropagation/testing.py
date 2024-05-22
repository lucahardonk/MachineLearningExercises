import os
import cv2
import numpy as np
import matplotlib as plot
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


myModel = 'my_handwritten_model' # salvato in questa directory

model = tf.keras.models.load_model(myModel)


mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
x_test  = tf.keras.utils.normalize(x_test, axis=1) # prendo solo il testing


# valutazione del modello passandogli i dati riservati per il testing
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy}")
print(f"Test loss: {loss}")




# cicla le immagini da me scritte

current_image_number = 0 
max_images = 9
image_directory = 'my_images'


while os.path.isfile(f"{image_directory}/image_{current_image_number}.png"):
    try:
        # Legge l'immagine dal percorso specificato e la converte in scala di grigi, mi serve un numpy array altrimenti ho un shape error
        img = cv2.imread(f"{image_directory}/image_{current_image_number}.png")[:,:,0] # 0 è la conversione implicita in scala di grigi
        # Inverte i colori dell'immagine
        img = np.invert(np.array([img]))
        # Esegue la predizione utilizzando il modello
        prediction = model.predict(img)
        # Stampa la previsione ottenuta
        print(f"La mia modesta previsione è: {np.argmax(prediction)}")
        # Visualizza l'immagine in una finestra con titolo "Image"
        cv2.imshow('Image', img[0])
        # Attende per 2000 millisecondi (2 secondi) per un tasto premuto
        cv2.waitKey(2000)
        # Chiude tutte le finestre OpenCV
        cv2.destroyAllWindows()
    except:
        # Se si verifica un errore durante l'esecuzione, stampa "errore"
        print("Errore")
    finally:
        # Incrementa il numero dell'immagine corrente
        current_image_number += 1