# Canny Edge Detection

This project implements the **Canny Edge Detection** algorithm to detect edges in an image. The workflow includes the following steps:

1. **Grayscale Conversion**: The image is converted to grayscale by applying a weighted average of the RGB values.
2. **Gaussian Blur**: A Gaussian blur is applied to smooth the image and reduce noise that may cause false edges.
3. **Gradient Calculation**: The Sobel operators are used to compute the gradients in both the x and y directions to determine the magnitude and direction of intensity changes.
4. **Non-Maximum Suppression**: The edges are thinned by retaining only the strongest pixels in the direction of the gradient.
5. **Double Thresholding and Hysteresis**: Strong edge pixels are marked, weak pixels are kept if connected to strong edges, and others are discarded.
6. **Output**: The resulting edge-detected image is saved and displayed.

## Usage

Run the script with the following command:

```bash
python CE_detection.py <input_image>

