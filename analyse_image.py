import h5py
import numpy as np

import torch
from torch import nn
from torchvision.models import MobileNetV2
from torchvision.ops.misc import Conv2dNormActivation
import torchvision.transforms.v2 as transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

def process_image(image, sat_prctile = 99):
    """
    Processes an image by normalizing it and then taking the square root.
    """

    sat = np.percentile(image, sat_prctile)
    # median filter
    image = sat * np.tanh(image / sat)
    image = image / np.sqrt(np.sum(image**2))
    return image

def select_patches(image, patch_size, num_patches, threshold):
    """
    Selects patches from an image based on the 99th percentile grey value threshold.
    """
    patches = []
    
    # find patches in random locations within the central 50% of the image
    x_min = image.shape[0] // 4
    x_max = image.shape[0] * 3 // 4
    y_min = image.shape[1] // 4
    y_max = image.shape[1] * 3 // 4
    for _ in range(num_patches):
        while True:
            # Randomly select a patch
            x = np.random.randint(x_min, x_max - patch_size)
            y = np.random.randint(y_min, y_max - patch_size)
            patch = image[x:x+patch_size, y:y+patch_size]

            # Check if the patch meets the threshold criterion
            if np.percentile(patch, 99) > threshold:
                patches.append((x, y))
                break


    return patches

model = MobileNetV2()
model.features[0][0] = Conv2dNormActivation(1, 32, kernel_size=1, norm_layer=None, activation_layer=nn.ReLU)
model.classifier = nn.Sequential(

    nn.Dropout(0.4),
    nn.Linear(1280, 3),
    #nn.ReLU(inplace=True),
    #nn.Linear(100, 3),
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
checkpoint_path = '/home/asaph/src/autofocus/model_checkpoint_epoch_700.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

model.eval()
mean_file_path = '/media/gf/DBIO_Bianco_Lab7/ASAPH/231221/231221f3_plane2_power15_spacing0_5/mean_stack_231221f3_plane2_power15_spacing0_5.mat'
with h5py.File(mean_file_path, 'r') as file:
    print(f'loading {mean_file_path}')
    images = file['x'][:]  # Adjust the key based on your data file

dists = []
ms = []
writer = FFMpegWriter(fps=15)
fig = plt.figure(facecolor='0.9', figsize=(14, 14), dpi=100)
gs = fig.add_gridspec(nrows=9, ncols=9, left=0.05, right=0.85,
                      hspace=0.1, wspace=0.1)
ax0 = fig.add_subplot(gs[0:8, 0:7])
ax1 = fig.add_subplot(gs[0:4, 7:])
offset = 4
real_focus = []
img_range = range(5,55)
with writer.saving(fig, "focus_video.mp4", 100):

    for img in img_range:
        print(img)
        this_image = images[img, :, :]
        ax0.clear()
        ax0.imshow(this_image, interpolation='nearest', vmin=0, vmax=50, cmap='gray')
        # remove axis ticks
        ax0.set_xticks([])
        ax0.set_yticks([])
        real_focus.append((img - images.shape[0] // 2) * 0.5  + offset) # Assuming the distance between images is 0.5 microns

        patch_size = 250
        selected_patches = select_patches(this_image, patch_size, 128, 25)

        distance = []
        focus_class = []
        all_patches = []
        for i, patch in enumerate(selected_patches):
            this_patch = this_image[patch[0]:patch[0]+patch_size, patch[1]:patch[1]+patch_size].astype(np.float32)
            this_patch = this_patch.reshape(this_patch.shape[0] // 2, 2, this_patch.shape[1] // 2, 2).sum(axis=(1, 3))
            this_patch = process_image(this_patch, sat_prctile=95)
            all_patches.append(this_patch)
        all_patches = transforms.ToImage()(np.array(all_patches))
        all_patches = all_patches.permute(1, 2, 0).unsqueeze(1) 
        all_patches = all_patches.to(device)
        with torch.no_grad():
            pred = model(all_patches).cpu().detach().numpy()  # Get the predictions and convert to numpy

        mean_distance = np.mean(pred[:, 0] *40)

        mean_class = np.mean(np.argmax(pred[:, 1:], axis=1))
        if mean_class < 0.5:
            mean_distance *= -1
        dists.append(mean_distance)
        ax1.clear()
        ax1.plot(real_focus, label='Real', color='k', linewidth=3)
        ax1.plot(dists, label='Inferred', color='b', linewidth=2)
        ax1.axhline(y=real_focus[-1], color='r', linestyle='--')
        ax1.axhline(y=0, color='k', linestyle='--')
        plt.xlim(0, len(img_range))
        plt.ylim(-15, 15)
        ax1.set_xlabel('Image Index', fontsize=16)
        ax1.set_ylabel('Imaging to Focal Distance (microns)', fontsize=16)
        ax1.legend(fontsize=14, loc='lower right')
        # set ticks font size
        ax1.tick_params(axis='both', which='major', labelsize=14)


        writer.grab_frame()
    writer.finish()





