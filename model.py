import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

def extract_features(image_path):
    try:
        # Read the preprocessed image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        
        if image is not None:
            # Flatten the pixel values
            features = image.flatten()
            return features
        else:
            print(f"Failed to read image: {image_path}")
            return None
    except Exception as e:
        print(f"Error processing image: {image_path}\n{str(e)}")
        return None

def load_dataset(dataset_folder):
    features_list = []
    labels_list = []
    
    label_encoder = LabelEncoder()
    
    for class_label in os.listdir(dataset_folder):
        class_path = os.path.join(dataset_folder, class_label)
        if os.path.isdir(class_path):
            for image_filename in os.listdir(class_path):
                if image_filename.lower().endswith((".jpg", ".png")):
                    image_path = os.path.join(class_path, image_filename)
                    features = extract_features(image_path)
                    if features is not None:
                        features_list.append(features)
                        labels_list.append(class_label)

    labels_encoded = label_encoder.fit_transform(labels_list)
    
    return features_list, labels_encoded

def train_knn_classifier(X, y, n_neighbors=3):
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(X, y)
    return knn_classifier

if __name__ == "__main__":
    # Set the path to the dataset folder containing preprocessed images
    dataset_folder = r"C:\Leo PBL\Dataset\preprocessed"

    if not os.path.exists(dataset_folder):
        print(f"Dataset folder does not exist: {dataset_folder}")
    else:
        # Initialize lists to store features and labels
        features, labels = load_dataset(dataset_folder)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Train K-NN classifier
        knn_classifier = train_knn_classifier(X_train, y_train, n_neighbors=3)

        # Save the trained model to a file
        model_filename = r"C:\Leo PBL\Code\model.pkl"
        joblib.dump(knn_classifier, model_filename)

        # Make predictions on the test data
        y_pred = knn_classifier.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")
