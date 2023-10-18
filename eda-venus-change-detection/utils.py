import matplotlib.pyplot as plt


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

    # Create a plot with three subplots
    plt.figure(figsize=(12, 4))

    # Plot the first image
    plt.subplot(1, 3, 1)
    plt.imshow(image1)
    plt.title("Image 1")
    plt.axis("off")

    # Plot the second image
    plt.subplot(1, 3, 2)
    plt.imshow(image2)
    plt.title("Image 2")
    plt.axis("off")

    # Plot the mask
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap="gray")
    plt.title(img_id)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def plot_batch(batch):
    # Calculate the number of samples in the batch
    batch_size = batch['image1'].shape[0]

    # Determine the number of rows and columns for subplots
    num_rows = batch_size
    num_cols = 3  # You can adjust this depending on your needs (e.g., for larger batches)

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))

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
        axes[i, 2].set_title(img_id)
        axes[i, 2].axis('off')

    plt.tight_layout()
    return fig
