import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.externals import joblib

# Step 1: Extract background color distribution
def extract_background_color(image):
    # Assuming the corners of the image represent background
    corners = [(0, 0), (0, image.shape[0]-1), (image.shape[1]-1, 0), (image.shape[1]-1, image.shape[0]-1)]
    background_pixels = []
    for corner in corners:
        background_pixels.append(image[corner[1], corner[0]])
    return np.mean(background_pixels, axis=0)

# Step 2: Extract obfuscating lines
def extract_obfuscating_lines(image, erosion_param):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply erosion to thin out lines
    kernel = np.ones((erosion_param, erosion_param), np.uint8)
    eroded_image = cv2.erode(gray_image, kernel, iterations=1)
    
    # Thresholding to extract obfuscating lines
    _, obfuscating_lines = cv2.threshold(eroded_image, 1, 255, cv2.THRESH_BINARY)
    return obfuscating_lines

# Step 3: Segment image into pieces
def segment_image(image, background_color, num_chars):
    # Convert image to HSV format for easier shade identification
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Thresholding to identify background pixels
    background_threshold = 30  # Adjust this threshold based on the image
    background_mask = cv2.inRange(hsv_image, (0, 0, 0), background_color + background_threshold)
    
    # Perform clustering on non-background pixels to identify characters
    non_background_pixels = hsv_image[np.where(background_mask == 0)]
    normalized_pixels = preprocessing.normalize(non_background_pixels)
    
    kmeans = KMeans(n_clusters=num_chars)
    kmeans.fit(normalized_pixels)
    labels = kmeans.labels_
    
    # Find vertical columns with few non-background pixels to further refine segmentation
    column_threshold = 10  # Adjust this threshold based on the image
    column_counts = np.sum(background_mask == 0, axis=0)
    char_indices = np.where(column_counts > column_threshold)[0]
    
    char_segments = []
    for i in range(len(char_indices)-1):
        char_segments.append(image[:, char_indices[i]:char_indices[i+1]])
    
    return char_segments

# Step 4: Use ML models to identify the char in each piece
def identify_char(char_segment):
    # Preprocess char_segment (e.g., remove colors, trim blank space)
    # Implement your own logic here based on the requirements
    
    # Load trained ML model for char identification
    model = joblib.load('char_model.pkl')  # Replace with your trained model filename
    
    # Classify the char segment using the ML model
    char_label = model.predict(char_segment)
    
    return char_label

# Load and process the image
image = cv2.imread('hidden_number_image.jpg')  # Replace with your image filename

background_color = extract_background_color(image)
obfuscating_lines = extract_obfuscating_lines(image, erosion_param=5)  # Adjust erosion_param as needed
char_segments = segment_image(image, background_color, num_chars=5)  # Adjust num_chars based on the image

# Identify the char in each segment
for char_segment in char_segments:
    char_label = identify_char(char_segment)
    print("Identified char: ", char_label)

