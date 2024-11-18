#%%
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# image = cv2.imread("dataset/")

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('origin image')
# plt.imshow(image, cmap='gray')

# gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('grqy image')
# plt.imshow(gray_scale, cmap='gray')
# intial visualization of images shows the existing of noise
# noise removal in needed for all given images

# %%

def noise_removal_images(input_path: str, output_path: str, kernel_size: int = 5,denoising_method: str="median"):
    """
    Reads all images in a given directory, Apply denoising and then save results.

        Parameters:
        input_path (str): Path to the directory containing the images.
        output_path (str): Path to save the processed images.
        kernel_size (int): Kernel size: size of the square shaped window 
        denoising_method (str): Denoising method, "median" or "gaussian".
    """

    # Here median blur is used due to good results for biological images
    # Kernel size set by default to 5 as it gives balance between noise reduction and detail presevation

    # First let's ensure the output path exists
    os.makedirs(output_path, exist_ok=True)

    # read images
    images = [file for file in os.listdir(input_path) if file.lower().endswith(('.jpg'))]

    for image_file in images:
        # read the image
        image_path = os.path.join(input_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Skipping invalid image: {image_file}")
            continue

        # apply noise removal
        if denoising_method == "median":
            denoised = cv2.medianBlur(image, kernel_size)
        elif denoising_method == "gaussian":
            denoised = cv2.GaussianBlur(image,(kernel_size, kernel_size), 0)
        else:
            raise ValueError("Unrecognized denoising method.")
        
        # save output image
        output_file_path = os.path.join(output_path, f"denoised_{image_file}")
        cv2.imwrite(output_file_path, denoised)

#noise_removal_images("dataset/", "dataset/denoisedK3/", 3)

def contrast_enhancement(input_path: str, output_path: str, method: str="CLAHE"):
    """
    Reads all images in a directory, applies contrast enhancement, and saves the results.

     Parameters:
        input_path (str): Path to the directory containing the images.
        output_path (str): Path to save the processed images.
        method (str): Contrast enhancement method, "CLAHE" or "histo" (Histogram Equalization).
    """
    # By default contrast method is set to CLAHE as it gives better results with biological images due to localized enhancement

    # First let's ensure the output path exists
    os.makedirs(output_path, exist_ok=True)

    # read images
    images = [file for file in os.listdir(input_path) if file.lower().endswith(('.jpg'))]

    for image_file in images:
        # read the image
        image_path = os.path.join(input_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Skipping invalid image file: {image_file}")
            continue

        # apply contrast enhancement
        if method == "CLAHE":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        elif method == "histo":
            enhanced = cv2.equalizeHist(image)
        else:
            raise ValueError("Unrecognized contrast enhancement method.")
        
        # save result
        output_file_path = os.path.join(output_path, f"enhanced_{method}_{image_file}")
        cv2.imwrite(output_file_path, enhanced)

#contrast_enhancement("dataset/denoised/", "dataset/contrast enhanced/")

def background_removal(input_path: str, output_path: str, kernel_size: int=15, method="substract"):
    """
    Removes the background from all images in a directory using morphological operations.

    Parameters:
        input_path (str): Path to the directory containing the images.
        output_path (str): Path to save the background-removed images.
        kernel_size (tuple): Size of the kernel used for morphological operations.
        method (str): Background removal method, "subtract" or "divide".
                      - "subtract": Subtracts the background.
                      - "divide": Divides the image by the background.
    """

    # By default kernel size (tuple) is set to 15x15 as that will give better results for petri dishes images
    # By default substract method is used as it locates colonnies better

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # read images
    images = [file for file in os.listdir(input_path) if file.lower().endswith(('.jpg'))]

    for image_file in images:
        # read the image
        image_path = os.path.join(input_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Skipping invalid image file: {image_file}")
            continue
        
        # Create a kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Estimate the background using morphological closing
        background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        # apply background removal 
        if method == "substract":
            result = cv2.subtract(background, image)
        elif method == "divide":
            # Normalize to avoid division by zero
            background = np.where(background == 0, 1, background)
            result = cv2.divide(image, background, scale=255)
        else:
            raise ValueError("Unrecognized background removal method.")   
        
         # Normalize the result to range [0, 255]
        result = cv2.normalize(result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # save result
        output_file_path = os.path.join(output_path, f"bg_removed_{image_file}")
        cv2.imwrite(output_file_path, result)

#background_removal("dataset/contrast enhanced/", "dataset/background removal/")

def normalizing_images(input_path: str, output_path: str):
    """
    Standardizes image intensity for consistency

    Parameters:
        input_path (str): Path to the directory containing the images.
        output_path (str): Path to save the background-removed images.

    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # read images
    images = [file for file in os.listdir(input_path) if file.lower().endswith(('.jpg'))]

    for image_file in images:
        # read the image
        image_path = os.path.join(input_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Skipping invalid image file: {image_file}")
            continue

        
        # Normalize pixel values to range [0, 255]
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # save result
        output_file_path = os.path.join(output_path, f"bg_removed_{image_file}")
        cv2.imwrite(output_file_path, image)

normalizing_images("dataset/background removal/", "dataset/normalized/")
