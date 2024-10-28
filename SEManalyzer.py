import cv2
import numpy as np
import statistics
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# Specify the image file path
image_path = './Data/sem_image.bmp'  # TODO: Change this to your actual image file path
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Cannot load the image. Please check the path: {image_path}")
    exit()

# Initialize global variables
scale_roi = None

# Callback function for ROI selection
def line_select_callback(eclick, erelease):
    global scale_roi
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    scale_roi = (min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2))

# Function to toggle the RectangleSelector
def toggle_selector(event):
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print('ROI selection completed.')
        toggle_selector.RS.set_active(False)
        plt.close()

# Convert the image to RGB for displaying with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Set up the matplotlib figure and axes
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.imshow(image_rgb)
ax.set_title('Drag to select the scale bar area, then press "q" to confirm.')

# Initialize the RectangleSelector
toggle_selector.RS = RectangleSelector(
    ax, line_select_callback,
    drawtype='box', useblit=True,
    button=[1],  # Use left mouse button only
    minspanx=5, minspany=5,
    spancoords='pixels',
    interactive=True
)

# Connect the toggle function to key press events
plt.connect('key_press_event', toggle_selector)
plt.show()

# Exit if no ROI was selected
if scale_roi is None:
    print("No ROI selected. Exiting the program.")
    exit()

x_roi, y_roi, w_roi, h_roi = scale_roi

# Crop the selected scale bar area
scale_bar = image[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]

# Calculate the pixel length of the scale bar
gray_bar = cv2.cvtColor(scale_bar, cv2.COLOR_BGR2GRAY)
_, thresh_bar = cv2.threshold(gray_bar, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours in the thresholded scale bar image
contours, _ = cv2.findContours(thresh_bar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_length = 0
for cnt in contours:
    x_bar, y_bar, w_bar, h_bar = cv2.boundingRect(cnt)
    if w_bar > max_length:
        max_length = w_bar  # Update the maximum length

print(f"Pixel length of the scale bar: {max_length} pixels")

# Input the actual length and unit of the scale bar
scale_value = float(input("Enter the actual length of the scale bar (numeric value only): "))
scale_unit = input("Enter the unit of the scale bar (e.g., µm, nm): ")

# Calculate pixels per unit length
pixels_per_unit = max_length / scale_value

# Convert the main image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# Threshold the blurred image (automatic threshold using Otsu's method)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# For manual threshold adjustment, uncomment and set the threshold value:
# thresh_value = 127  # Adjust between 0 and 255 as needed
# _, thresh = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY_INV)

# Apply morphological operations to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Find contours in the processed image
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter spherical particles and calculate their diameters
diameters = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 10:  # Modified area threshold
        # Obtain the minimum enclosing circle of the contour
        (x_cnt, y_cnt), radius = cv2.minEnclosingCircle(cnt)
        # Calculate the circularity of the contour
        circularity = (4 * np.pi * area) / (cv2.arcLength(cnt, True) ** 2)
        if circularity > 0.2:  # Modified circularity threshold
            diameter = 2 * radius
            diameters.append(diameter)

# Convert diameters from pixels to actual units
real_diameters = [d / pixels_per_unit for d in diameters]

# Calculate and display the results
count = len(real_diameters)
if count > 0:
    mean_diameter = statistics.mean(real_diameters)
    std_dev = statistics.stdev(real_diameters) if count > 1 else 0

    print(f"Number of particles: {count}")
    print(f"Average diameter: {mean_diameter:.2f} {scale_unit}")
    print(f"Diameter standard deviation: {std_dev:.2f} {scale_unit}")
else:
    print("No particles detected.")

# Visualize the detected particles (optional)
# Create a copy of the original image
output_image = image.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 10:  # Modified area threshold
        (x_cnt, y_cnt), radius = cv2.minEnclosingCircle(cnt)
        circularity = (4 * np.pi * area) / (cv2.arcLength(cnt, True) ** 2)
        if circularity > 0.2:  # Modified circularity threshold
            center = (int(x_cnt), int(y_cnt))
            radius = int(radius)
            cv2.circle(output_image, center, radius, (0, 255, 0), 2)

# Display the output image with detected particles
cv2.imshow('Detected Particles', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
