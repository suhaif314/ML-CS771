import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Identify background pixels
def extract_background_color(image):
    corners = [
        image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]
    ]
    background_color = np.mean(corners, axis=(0, 1))
    return background_color

# Step 2: Identify pixels that belong to obfuscating lines
def eliminate_obfuscating_lines(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply erosion to thin out lines
    eroded = cv2.erode(gray, kernel=np.ones((3, 3), np.uint8), iterations=2)
    
    # Threshold to obtain binary image
    _, binary = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Invert binary image
    inverted = cv2.bitwise_not(binary)
    
    # Apply inverted mask to original image
    removed_lines = cv2.bitwise_and(image, image, mask=inverted)
    
    return removed_lines

# Step 3: Segment image into pieces
def segment_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Threshold to obtain binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    # Extract segmented pieces
    segments = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        segment = image[y:y+h, x:x+w]
        segments.append(segment)
    
    return segments

# Step 4: Use ML models to identify the char in each piece
def preprocess_piece(piece):
    # Convert to grayscale
    gray = cv2.cvtColor(piece, cv2.COLOR_RGB2GRAY)
    
    # Trim the piece to contain mostly the char
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    trimmed_piece = piece[y:y+h, x:x+w]
    
    # Rotate the piece to match reference image rotations
    rotations = [0, 10, -10, 20, -20, 30, -30]
    best_match = None
    best_similarity = -1
    
    for rotation in rotations:
        rotated_piece = ndimage.rotate(trimmed_piece, rotation, reshape=False)
        processed_rotated_piece = cv2.cvtColor(rotated_piece, cv2.COLOR_RGB2GRAY).flatten()
        
        for reference_image in reference_images:
            similarity = np.mean(processed_rotated_piece == reference_image)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = reference_image
    
    return best_match

# Load reference images for classification
reference_images = []
for i in range(16):
    reference_image = cv2.imread(f"C:\Users\DELL\AppData\Local\Programs\Python\Python311\assign2_Cs771\assn2\reference/{i}.png")
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY).flatten()
    reference_images.append(reference_image)

# Main algorithm
def extract_parity(image):
    background_color = extract_background_color(image)
    removed_lines = eliminate_obfuscating_lines(image)
    segments = segment_image(removed_lines)
    
    parity = ""
    
    for segment in segments:
        processed_segment = preprocess_piece(segment)
        predicted_background_color = classifier.predict([processed_segment])[0]
        if np.allclose(predicted_background_color, background_color):
            parity += "0"
        else:
            parity += "1"
    
    return int(parity, 2) % 2  # Convert binary to decimal and calculate parity

# Train the classifier
def train_classifier():
    X = []  # Features
    y = []  # Labels

    # Load training images and labels
    for i in range(1, 101):
        image_path = f"C:\Users\DELL\AppData\Local\Programs\Python\Python311\assign2_Cs771\assn2\train/{i}.png"
        image = cv2.imread(image_path)
        background_color = extract_background_color(image)
        removed_lines = eliminate_obfuscating_lines(image)
        segments = segment_image(removed_lines)
        
        for segment in segments:
            processed_segment = preprocess_piece(segment)
            X.append(processed_segment)
            y.append(background_color)
    
    # Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the classifier
    classifier = SVC()
    classifier.fit(X_train, y_train)
    
    # Evaluate the classifier
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))

    return classifier

# Example usage
image_path = r"0.png"
image = cv2.imread(image_path)
classifier = train_classifier()
parity = extract_parity(image)
print("Parity:", parity)
