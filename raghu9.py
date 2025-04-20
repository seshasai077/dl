import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, Flatten, Dense 
import matplotlib.pyplot as plt
vocab_size = 10000 
max_length = 100  
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size) 
x_train = pad_sequences(x_train, maxlen=max_length, padding='post') 
x_test = pad_sequences(x_test, maxlen=max_length, padding='post')
embedding_dim = 16  
model = Sequential([ 
Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
input_length=max_length), 
Flatten(),  
Dense(16, activation='relu'), 
Dense(1, activation='sigmoid')  
]) 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
model.summary() 
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, 
y_test)) 
embedding_layer = model.layers[0] 
weights = embedding_layer.get_weights()[0] 
print("Shape of the embedding matrix:", weights.shape) 
test_loss, test_acc = model.evaluate(x_test, y_test) 
print(f"Test Accuracy: {test_acc:.4f}")
plt.plot(history.history['accuracy'], label='Training Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.legend() 
plt.show()