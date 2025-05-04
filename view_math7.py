import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load and display the math7 image
def view_math7():
    image_path = "uploads/math7.jpeg"
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert from BGR to RGB for matplotlib
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.title("Math7 Image")
    plt.axis('off')
    
    # Save the output
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, "math7_display.jpg")
    plt.savefig(output_path)
    print(f"Saved image to {output_path}")
    
    # Show the image
    plt.show()
    
    return image

# Run the simple test
if __name__ == "__main__":
    view_math7()
