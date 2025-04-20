
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import numpy as np 
import matplotlib.pyplot as plt 
vocab_size = 10000  
max_length = 200   
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size) 
x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post') 
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post') 
model = keras.Sequential([ 
keras.layers.Embedding(input_dim=vocab_size, output_dim=128, 
input_length=max_length), 
keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2), 
keras.layers.Dense(64, activation='relu'), 
keras.layers.Dropout(0.5), 
keras.layers.Dense(1, activation='sigmoid')]) 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
model.summary() 
history = model.fit(x_train, y_train, epochs=2, batch_size=64, validation_data=(x_test, y_test)) 
test_loss, test_acc = model.evaluate(x_test, y_test) 
print(f"Test Accuracy: {test_acc:.4f}") 
new_review = [[1, 45, 6, 200, 54]]  
new_review_sequence = pad_sequences(new_review, maxlen=max_length) 
prediction = model.predict(new_review_sequence) 
print("Positive" if prediction > 0.5 else "Negative") 
plt.figure(figsize=(12, 4)) 
plt.subplot(1, 2, 1) 
plt.plot(history.history['accuracy'], label='Train Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.legend() 
plt.title('Model Accuracy') 
plt.subplot(1, 2, 2) 
plt.plot(history.history['loss'], label='Train Loss') 
plt.plot(history.history['val_loss'], label='Validation Loss') 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.legend() 
plt.title('Model Loss')
plt.show()