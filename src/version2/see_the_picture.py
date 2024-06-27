'''
Test on function of crop_with_points by using the points, and I can see the result image.
最終是用以上兩點來切分
(79, 57), (575, 426)
左上和右下
'''
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
def crop_with_points(image_path):
    """
    Crop an image based on specified points.

    Args:
    - image_path (str): Path to the image file.

    Returns:
    - cropped_img (PIL.Image): Cropped image based on the provided points.
    """
    points = [(79, 57), (575, 428), (575, 57), (79, 426)]
    # Load the image
    img = Image.open(image_path)
    print(img.size)
    # Define the four points
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]

    # Find the bounding box for cropping
    left = min(x1, x4)
    upper = min(y1, y2)
    right = max(x2, x3)
    lower = max(y3, y4)
    print(left, upper, right,lower)
    # Crop the image
    cropped_img = img.crop((left, upper, right, lower))

    return cropped_img
def visualize_transform(image_folder_path):
    image_paths = [os.path.join(image_folder_path, filename) for filename in os.listdir(image_folder_path)]

    # Define a transform to resize and convert images to tensors
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to 128x128
        transforms.ToTensor()
    ])

    for path in image_paths:
        # Load image using PIL
        image_pil = crop_with_points(path).convert('RGB')  # Ensure it's RGB
        print(image_pil.size)
        # Apply transformation
        image_tensor = transform(image_pil)
        print(image_tensor)
        # Convert tensor back to image for visualization (optional)
        transformed_image = transforms.ToPILImage()(image_tensor)

        # Display the original and transformed images
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image_pil)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(transformed_image)
        axes[1].set_title('Transformed Image')
        axes[1].axis('off')
        plt.show()

if __name__ == "__main__":
    # Specify the path to your image folder
    image_folder_path = r"C:\Users\User\Desktop\temp"
    # Visualize the transformation
    visualize_transform(image_folder_path)
