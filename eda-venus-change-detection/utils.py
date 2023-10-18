import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_sample(dataset, idx):
    sample = dataset[idx]

    # Extract the images and mask
    image1 = sample["image1"]
    image2 = sample["image2"]
    mask = sample["mask"]
    img_id = sample["img_id"]

    # Convert images and mask to numpy arrays
    image1 = image1.permute(1, 2, 0).cpu().numpy()  # Assuming image1 is a torch tensor
    image2 = image2.permute(1, 2, 0).cpu().numpy()  # Assuming image2 is a torch tensor
    mask = mask.squeeze().cpu().numpy()  # Assuming mask is a torch tensor

    # Create an overlay of the mask on image2
    overlay = image2.copy()
    overlay[mask == 1] = [1, 0, 0]  # Change positive mask to red
    
    # Mix image2 and overlay to get semi-transparent effect
    alpha = 0.5  # Transparency factor: adjust as needed
    image2_overlay = np.clip(image2 * (1 - alpha) + overlay * alpha, 0, 1)

    # Create a plot with four subplots
    plt.figure(figsize=(16, 4))

    # Plot the first image
    plt.subplot(1, 4, 1)
    plt.imshow(image1)
    plt.title("Image 1")
    plt.axis("off")

    # Plot the second image
    plt.subplot(1, 4, 2)
    plt.imshow(image2)
    plt.title("Image 2")
    plt.axis("off")

    # Plot the mask
    plt.subplot(1, 4, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")
    
    # Plot image2 with overlay
    plt.subplot(1, 4, 4)
    plt.imshow(image2_overlay)
    plt.title("Image 2 with Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def plot_batch(batch):
    # Calculate the number of samples in the batch
    batch_size = batch['image1'].shape[0]

    # Determine the number of rows and columns for subplots
    num_rows = batch_size
    num_cols = 4  # Updated to account for the overlay

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4 * num_rows))

    for i in range(batch_size):

        # Extract the images and mask
        image1 = batch['image1'][i, :, :, :]
        image2 = batch['image2'][i, :, :, :]
        mask = batch['mask'][i, :, :, :]
        img_id = batch['img_id'][i]

        # Convert images and mask to numpy arrays
        image1 = image1.permute(1, 2, 0).cpu().numpy()  # Assuming image1 is a torch tensor
        image2 = image2.permute(1, 2, 0).cpu().numpy()  # Assuming image2 is a torch tensor
        mask = mask.squeeze().cpu().numpy()  # Assuming mask is a torch tensor

        # Create an overlay of the mask on image2
        overlay = image2.copy()
        overlay[mask == 1] = [1, 0, 0]  # Change positive mask to red

        # Mix image2 and overlay to get semi-transparent effect
        alpha = 0.5  # Transparency factor
        image2_overlay = np.clip(image2 * (1 - alpha) + overlay * alpha, 0, 1)

        # Plot the first image
        axes[i, 0].imshow(image1)
        axes[i, 0].set_title('Image 1')
        axes[i, 0].axis('off')

        # Plot the second image
        axes[i, 1].imshow(image2)
        axes[i, 1].set_title('Image 2')
        axes[i, 1].axis('off')

        # Plot the mask
        axes[i, 2].imshow(mask, cmap='gray')
        axes[i, 2].set_title('Mask')
        axes[i, 2].axis('off')

        # Plot image2 with overlay
        axes[i, 3].imshow(image2_overlay)
        axes[i, 3].set_title('Image 2 with Overlay')
        axes[i, 3].axis('off')

    plt.tight_layout()
    return fig

def plot_with_predictions(batch, model):
    # Ensure the model is in eval mode
    model.eval()

    # Calculate the number of samples in the batch
    batch_size = batch['image1'].shape[0]
    num_rows = batch_size
    num_cols = 5  # For Image1, Image2, Mask, Overlay, Prediction

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))

    with torch.no_grad():  # Ensure no gradients are computed
        input1 = batch['image1']
        input2 = batch['image2']
        predictions = model(input1.float(), input2.float()).squeeze().cpu().numpy()

    for i in range(batch_size):

        # Extract the images and mask
        image1 = batch['image1'][i].permute(1, 2, 0).cpu().numpy()
        image2 = batch['image2'][i].permute(1, 2, 0).cpu().numpy()
        mask = batch['mask'][i].squeeze().cpu().numpy()

        overlay = image2.copy()
        overlay[mask == 1] = [255, 0, 0]  # red for positive change

        # Plot the first image
        axes[i, 0].imshow(image1)
        axes[i, 0].set_title('Image 1')
        axes[i, 0].axis('off')

        # Plot the second image
        axes[i, 1].imshow(image2)
        axes[i, 1].set_title('Image 2')
        axes[i, 1].axis('off')

        # Plot the mask
        axes[i, 2].imshow(mask, cmap='gray')
        axes[i, 2].set_title('Mask')
        axes[i, 2].axis('off')

        # Plot the overlay
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')

        # Plot the model prediction
        axes[i, 4].imshow(predictions[i], cmap='gray')
        axes[i, 4].set_title('Prediction')
        axes[i, 4].axis('off')

    plt.tight_layout()
    plt.show()
