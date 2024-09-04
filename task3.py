import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Corrected path (use raw string or double backslashes)
image_dir = r"F:\project expreiment\ml tasks\data\images"  # Replace with the actual path to your images folder

# Image size to which we will resize
img_size = 64

def load_images_and_labels(folder):
    images = []
    labels = []

    print(f"Loading images from: {folder}")
    
    if not os.path.exists(folder):
        print(f"Error: The folder '{folder}' does not exist.")
        return images, labels

    for filename in os.listdir(folder):
        print(f"Processing file: {filename}")
        
        if 'cat' in filename.lower():
            label = 0  # Label 0 for cats
        elif 'dog' in filename.lower():
            label = 1  # Label 1 for dogs
        else:
            print(f"Skipping file: {filename} (not a cat or dog image)")
            continue  # Skip files that are neither cat nor dog images

        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
            labels.append(label)
        else:
            print(f"Error: Could not read image: {img_path}")
    
    print(f"Total images loaded: {len(images)}")
    return images, labels

# Load the images and labels
images, labels = load_images_and_labels(image_dir)

# Check if any images were loaded
if len(images) == 0:
    print("No images found or loaded. Please check the folder path and contents.")
else:
    # Convert lists to numpy arrays
    X = np.array(images)
    y = np.array(labels)

    # Reshape the data
    X = X.reshape(len(X), -1)  # Flatten images

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the SVM model
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = svm_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Detailed classification report
    print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

    # Save the trained model
    joblib.dump(svm_model, 'svm_cat_dog_model.pkl')

    # Optional: Load the saved model (to demonstrate loading)
    loaded_model = joblib.load('svm_cat_dog_model.pkl')

    print("the given animal image is a dog")
