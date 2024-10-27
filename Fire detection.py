import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import os

# Step 1: Feature Extraction Function
def extract_color_features(image):
    """
    Extracts color features (percentage of red, green, and blue pixels) for fire detection.
    """
    # Resize image for consistency
    image = cv2.resize(image, (64, 64))
    
    # Convert image to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Calculate the proportion of each color in the image
    red_pixels = np.sum((image_rgb[:, :, 0] > 150) & (image_rgb[:, :, 1] < 100) & (image_rgb[:, :, 2] < 100))
    yellow_pixels = np.sum((image_rgb[:, :, 0] > 150) & (image_rgb[:, :, 1] > 150) & (image_rgb[:, :, 2] < 100))
    total_pixels = image_rgb.shape[0] * image_rgb.shape[1]
    
    red_ratio = red_pixels / total_pixels
    yellow_ratio = yellow_pixels / total_pixels
    
    return [red_ratio, yellow_ratio]

# Step 2: Load Dataset and Extract Features
fire_images_path = "path/to/fire_images"
non_fire_images_path = "path/to/non_fire_images"

X = []
y = []

# Load fire images
for img_file in os.listdir(fire_images_path):
    img_path = os.path.join(fire_images_path, img_file)
    image = cv2.imread(img_path)
    if image is not None:
        features = extract_color_features(image)
        X.append(features)
        y.append(1)  # Label 1 for fire

# Load non-fire images
for img_file in os.listdir(non_fire_images_path):
    img_path = os.path.join(non_fire_images_path, img_file)
    image = cv2.imread(img_path)
    if image is not None:
        features = extract_color_features(image)
        X.append(features)
        y.append(0)  # Label 0 for non-fire

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Step 3: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train an SVM Classifier
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
