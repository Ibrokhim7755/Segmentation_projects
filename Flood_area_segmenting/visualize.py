import matplotlib.pyplot as plt
'''
This code visualizes the results for original images, ground truth and prediction



'''


def tensor_2_im(t): 
    return (t * 255).detach().cpu().permute(2, 1, 0).numpy().astype(np.uint8)


import random

indices = random.sample(range(len(images)), 10)
fig, axs = plt.subplots(10, 3, figsize=(10, 30))

for i, idx in enumerate(indices):
    im = images[idx]
    pred = preds[idx]
    gt = lbls[idx]

    # Plot the original image
    axs[i, 0].imshow(tensor_2_im(im))
    axs[i, 0].axis('off')
    axs[i, 0].set_title('Original Image')

    # Plot the predicted label
    axs[i, 1].imshow(tensor_2_im((pred > 0.5).squeeze(0))[:, :, 1], cmap='gray')
    axs[i, 1].axis('off')
    axs[i, 1].set_title('Predicted Label')

    # Plot the ground truth label
    axs[i, 2].imshow(tensor_2_im(gt.unsqueeze(0)), cmap='gray')
    axs[i, 2].axis('off')
    axs[i, 2].set_title('Ground Truth Label')

plt.tight_layout()
plt.show()