import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split

# Dummy dataset setup
data_dir = 'dummy_data'  # Directory with images
csv_file = 'dummy_calorie_data.csv'  # CSV with calorie information

# Create dummy directory and files (you would have your own)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    
# Dummy images (create a couple of dummy images for demonstration)
dummy_images = [np.random.rand(224, 224, 3) for _ in range(5)]
for i, img in enumerate(dummy_images):
    cv2.imwrite(os.path.join(data_dir, f'image_{i}.jpg'), (img * 255).astype(np.uint8))

# Create dummy calorie data
calorie_data = {
    'image_name': [f'image_{i}.jpg' for i in range(5)],
    'calories': [100 + i * 50 for i in range(5)]
}
calorie_df = pd.DataFrame(calorie_data)
calorie_df.to_csv(csv_file, index=False)

# Load calorie data
calorie_df = pd.read_csv(csv_file)

# Preprocess images
image_size = (224, 224)
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Load and preprocess all images
image_paths = [os.path.join(data_dir, img) for img in os.listdir(data_dir)]
images = np.array([load_and_preprocess_image(img) for img in image_paths])

# Extract calorie values from the CSV
calories = calorie_df['calories'].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, calories, test_size=0.2, random_state=42)

# Load the ResNet50 model pre-trained on ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add new layers for our specific task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)

# Output layer for calorie estimation
predictions = Dense(1, activation='linear')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='mean_squared_error',  # For regression (calorie estimation)
              metrics=['mae'])  # Mean Absolute Error

# Data augmentation to avoid overfitting
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)

# Fit the model
batch_size = 2  # Reduced batch size for demonstration
epochs = 1  # Reduced epochs for demonstration

history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                    validation_data=(X_test, y_test),
                    epochs=epochs)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test Mean Absolute Error: {test_mae:.2f} calories')

# Save the trained model
model.save('food_calorie_estimator.h5')

# Load and preprocess a new image for prediction
new_image_path = os.path.join(data_dir, 'image_0.jpg')  # Use one of the dummy images
new_image = load_and_preprocess_image(new_image_path)
new_image = np.expand_dims(new_image, axis=0)  # Add batch dimension

# Predict the calorie content
predicted_calories = model.predict(new_image)
print(f'Predicted calories: {predicted_calories[0][0]:.2f}')
