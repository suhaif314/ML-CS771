import cv2
import numpy as np
import os
import multiprocessing

# from google.colab.patches import cv2_imshow
# from google.colab import drive
# drive.mount('/content/drive')


def remove_obfuscating_lines(image):
    # Convert the image to hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    # Find the minimum intensity value in the grayscale image
    min_val = np.min(v)

    para = 70

    # Set the threshold value
    threshold = min_val + para

    # Convert pixels lighter than the threshold to white
    image[v < threshold] = 255

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to turn colors other than white into black
    _, thresholded_image = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY_INV)

    return thresholded_image


def segment_fourth_character(image):
    # Check if the image is grayscale or color
    if len(image.shape) == 3:  # Color image
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Grayscale image
        gray = image

    # Apply adaptive thresholding to create a binary image
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by their areas in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Extract the bounding box and area of the fourth contour (index 3)
    x, y, w, h = cv2.boundingRect(contours[3])

    # Extract the segmented character using the bounding box
    character = image[y:y + h, x:x + w]

    return character


def process_image(image_path):
    image = cv2.imread(image_path)
    imag = remove_obfuscating_lines(image)
    img = segment_fourth_character(imag)
    return img


folder_path = r'\Users\DELL\AppData\Local\Programs\Python\Python311\assign2_Cs771\assn2\train'

# Get the list of image files
image_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)
               if filename.endswith('.jpg') or filename.endswith('.png')]

# Number of processes to use
num_processes = multiprocessing.cpu_count()

# Create a pool of processes
pool = multiprocessing.Pool(processes=num_processes)

# Process the images in parallel
results = pool.map(process_image, image_files)

# Close the pool
pool.close()
pool.join()

# Convert the list of images into a NumPy array
images_array = np.array(results)

# Display the result
cv2_imshow(images_array[1])
