import os
import cv2
import numpy as np
import matplotlib as plot
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


mnist = tf.keras.datasets.mnist                             #prendo le immagini dei digit
(x_train, y_train), (x_test, y_test) = mnist.load_data()    # divido il dataset in training data e validation data
x_train  = tf.keras.utils.normalize(x_train, axis=1)        # normalizzo pixel gryscale 0-255 -> 0-1
x_test  = tf.keras.utils.normalize(x_test, axis=1)          # = per x_test

model = tf.keras.models.Sequential()                        # Sequential è una sequenza lineare di neuroni
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))     # aggiungo un livello lineare converterno una matrice 28*28 in 784 neuroni in riga
model.add(tf.keras.layers.Dense(350, activation='relu'))    # classico collegamento dove ogni neurone è collegato a tutti i neuroni delle colonne che lo precedono e che lo susseguono
                                                            # composto da 128 neuroni e definendo la funzione di attivazione di ogni neurone, in questo caso una relu
model.add(tf.keras.layers.Dense(150, activation='relu'))    # ne immettiamo un altro livello uguale al precedente
model.add(tf.keras.layers.Dense(10, activation='softmax'))  # output layer the che applica racoglie tutti i parametri della rete neurale e applica una softmax (10 essendo digits da 0 a 9)
                                                            # in pratica mappa tutti i valori da 0 a 1 rislutando nella confidenza della preditione

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.21, momentum=0.9),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
'''
Adam: Adam è un algoritmo di ottimizzazione che combina le tecniche di Momentum e RMSProp, 
ed è comunemente usato per l'addestramento delle reti neurali perché è efficiente e spesso converge più velocemente rispetto ad altri ottimizzatori.

sparse_categorical_crossentropy: Questo parametro specifica la funzione di perdita da utilizzare durante l'addestramento del modello.
è una funzione di perdita comunemente usata per problemi di classificazione in cui le etichette sono rappresentate come interi (sparse) anziché codificate one-hot

accuracy: Questo parametro specifica le metriche da monitorare durante l'addestramento e la valutazione del modello
'''

# train il modello
model.fit(x_train, y_train, epochs=13) 

# salvataggio the modello
model.save('my_handwritten_model') 

# valutazione del modello passandogli i dati riservati per il testing
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy}\n")
print(f"Test loss: {loss}")