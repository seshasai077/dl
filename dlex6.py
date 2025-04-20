# Step 1: Import Required Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tkinter import filedialog, Tk

# Step 2: Load and Preprocess the Dataset
# Dataset should be structured as:
# dataset/
# ├── train/
# │   ├── cats/
# │   └── dogs/
# └── validation/
#     ├── cats/
#     └── dogs/

train_dir = r"C:\Users\sesha\Downloads\archive (1)\dataset\training_set"
validation_dir = r"C:\Users\sesha\Downloads\archive (1)\dataset\test_set"

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

print("✅ Class indices:", train_generator.class_indices)
index_to_label = {v: k for k, v in train_generator.class_indices.items()}

# Step 3: Build the CNN Model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(150, 150, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze the pre-trained layers

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification output
])

# Step 4: Compile and Train the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("🚀 Starting Model Training...")
history = model.fit(
    train_generator,
    epochs=1,
    validation_data=validation_generator
)
print("✅ Training Complete!")

# Step 5: Evaluate and Visualize Results
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"\n✅ Final Validation Accuracy: {val_accuracy * 100:.2f}%")

# Step 6: Function to Predict on New Images
def predict_multiple_images(model, image_paths):
    batch = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        batch.append(img_array)

    batch = np.array(batch)
    predictions = model.predict(batch)

    print("\n📋 Predictions:")
    for i, img_path in enumerate(image_paths):
        predicted_class = int(np.round(predictions[i][0]))
        label = index_to_label[predicted_class]
        confidence = predictions[i][0]

        print(f"🖼️ File: {os.path.basename(img_path)}")
        print(f"   - Prediction: {label}")
        print(f"   - Confidence: {confidence:.4f}")

        img = image.load_img(img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{os.path.basename(img_path)} ➜ {label} ({confidence:.2f})")
        plt.show()

# Step 7: Load Custom Images and Predict
root = Tk()
root.withdraw()
file_paths = filedialog.askopenfilenames(
    title="Select Images to Predict",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if file_paths:
    predict_multiple_images(model, file_paths)
else:
    print("❌ No images selected.")

# Step 8: Save the Trained Model
model.save("refined_dog_cat_classifier_with_transfer.h5")
print("✅ Model saved as refined_dog_cat_classifier_with_transfer.h5")
