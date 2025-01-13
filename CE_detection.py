import cv2
import numpy as np
import sys
import os
from matplotlib import pyplot as plt

# Convert image to grayscale
def to_grayscale(img):
    height, width, _ = img.shape
    grayscale_img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            r, g, b = img[i, j]
            grayscale_img[i, j] = 0.299*r + 0.587*g + 0.114*b
    return grayscale_img

# Apply Gaussian Blur to smooth the image and reduce noise
def gaussian_blur(img, kernel_size=5, sigma=1.4):
    def gaussian_kernel(size, sigma):
        k = np.linspace(-(size // 2), size // 2, size)
        kernel_1d = np.exp(-0.5 * (k/sigma)**2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        return kernel_2d

    kernel = gaussian_kernel(kernel_size, sigma)
    height, width = img.shape
    blurred_img = np.zeros_like(img)
    pad = kernel_size // 2
    padded_img = np.pad(img, pad, mode='constant')
    
    for i in range(height):
        for j in range(width):
            region = padded_img[i:i+kernel_size, j:j+kernel_size]
            blurred_img[i, j] = np.sum(region * kernel)
    
    return blurred_img

# Compute gradients using Sobel filters
def sobel_gradients(img):
    height, width = img.shape
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    gradient_magnitude = np.zeros_like(img, dtype=np.float32)
    gradient_direction = np.zeros_like(img, dtype=np.float32)
    
    padded_img = np.pad(img, 1, mode='constant')
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            gx = np.sum(sobel_x * padded_img[i-1:i+2, j-1:j+2])
            gy = np.sum(sobel_y * padded_img[i-1:i+2, j-1:j+2])
            gradient_magnitude[i, j] = np.sqrt(gx**2 + gy**2)
            gradient_direction[i, j] = np.arctan2(gy, gx)
    
    return gradient_magnitude, gradient_direction

# Non-Maximum Suppression to thin edges based on gradient direction
def non_max_suppression(magnitude, direction):
    rows, cols = magnitude.shape
    suppressed = np.zeros_like(magnitude)
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            q = 255
            r = 255
            
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                suppressed[i, j] = magnitude[i, j]
            else:
                suppressed[i, j] = 0
    
    return suppressed

# Double threshold and edge tracking by hysteresis
def threshold_and_hysteresis(img, low_threshold, high_threshold):
    strong = 255
    weak = 25 
    
    strong_i, strong_j = np.where(img >= high_threshold)
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))
    
    result = np.zeros_like(img)
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    
    rows, cols = img.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if result[i, j] == weak:
                # If weak pixel is connected to a strong one, promote it to strong
                if ((result[i+1, j-1:j+2] == strong).any() or 
                    (result[i-1, j-1:j+2] == strong).any() or 
                    (result[i, [j-1, j+1]] == strong).any()):
                    result[i, j] = strong
                else:
                    result[i, j] = 0
    
    return result


def canny_edge_detection(img, low_threshold=5, high_threshold=70, kernel_size=5):
    # Convert to grayscale
    gray_img = to_grayscale(img)
    
    # Apply Gaussian Blur with slightly increased kernel size
    blurred_img = gaussian_blur(gray_img, kernel_size)
    
    # Compute gradients
    gradient_magnitude, gradient_direction = sobel_gradients(blurred_img)
    
    # Non-Maximum Suppression
    thinned_edges = non_max_suppression(gradient_magnitude, gradient_direction)
    
    # Double thresholding and hysteresis
    final_edges = threshold_and_hysteresis(thinned_edges, low_threshold, high_threshold)
    
    return final_edges


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python CE_detection.py <input_image>")
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Unable to load image {input_image_path}")
        sys.exit(1)
    
    edges = canny_edge_detection(img)
    
    base_name = os.path.basename(input_image_path)
    name, ext = os.path.splitext(base_name)
    output_path = f"{name}_edge.png"
    cv2.imwrite(output_path, edges)

   
    plt.imshow(edges,cmap = 'gray')
    plt.title('input_image_edge'), plt.xticks([]), plt.yticks([])
    
    plt.show()
    
    print(f"Edge-detected image saved as: {output_path}")
