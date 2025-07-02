import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter, medfilt2d
import os
subdir = 'dat_grid_test'
# set constant seed for reproducibility
np.random.seed(0)

def process_image(image, sat_prctile = 99):
    """
    Processes an image by normalizing it and then taking the square root.
    """

    sat = np.percentile(image, sat_prctile)
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

def find_focus_dists(image_stack, distance_between_images, debug=False):
    """
    Finds the best focus image in a stack of images.
    """

    # Calculate contrast for each image
    uc = np.zeros(image_stack.shape[0])
    for i, image in enumerate(image_stack):
        # normalize image by L2 norm:
        image = process_image(image, sat_prctile=99)
        image = medfilt2d(image, 3)
        flt = (savgol_filter(np.mean(image, axis=0), 11, 3))
        uc[i] = -np.std(np.abs(np.fft.fft(flt))**2)

    best_focus = np.argmax(uc)
    if debug:
        fig, axs = plt.subplots(1, 8)
        for i, f in enumerate(range(-12, 16, 4)):
            axs[i].imshow(image_stack[best_focus+f], cmap='gray')
            axs[i].set_title(f'{f}')
        axs[7].plot(uc)
        axs[7].plot(savgol_filter(uc, 11, 3))
        plt.show()



        

    focus_dists = np.zeros(image_stack.shape[0])
    for i in range(image_stack.shape[0]):
        focus_dists[i] = (i - best_focus) * distance_between_images
    
    return best_focus, focus_dists

def extract_patches(mean_file_path, single_file_path, patch_size, num_patches, threshold):
    """
    Extracts patches from all images in a stack and returns the patches and a vector of distances.
    """
    with h5py.File(mean_file_path, 'r') as file:
        print(f'loading {mean_file_path}')
        mean_images = file['x'][:]  # Adjust the key based on your data file
    #with h5py.File(single_file_path, 'r') as file:
    #    print(f'loading {single_file_path}')
    #    raw_images = file['x'][:]  # Adjust the key based on your data file
    raw_images = mean_images
    #mean_images = raw_images
    # Select patches from the middle image
    middle_image = mean_images[mean_images.shape[0] // 2, :, :]
    #selected_patches = select_patches(middle_image, patch_size, num_patches, threshold)

    # loop over distances:
    file_names = []
    focal_distances = []
    file_identifier = os.path.basename(single_file_path).split('.')[0]
    for dist in range(-13, 13):
        selected_patches = select_patches(middle_image, patch_size, num_patches, threshold)

        for i, patch in enumerate(selected_patches):
            # find best focus image
            this_patch_mean = mean_images[:, patch[0]:patch[0]+patch_size, patch[1]:patch[1]+patch_size]
            this_patch_raw = raw_images[:, patch[0]:patch[0]+patch_size, patch[1]:patch[1]+patch_size]
            #print(f'finding best focus for patch {i}')
            best_focus, focus_dists = find_focus_dists(this_patch_mean, 0.5, debug=False)
            # make sure best focus is within reasonable range, ie within the middle 1/3 of the stack
            #print(f'best focus is {best_focus}')
            if best_focus < raw_images.shape[0] // 3 or best_focus > raw_images.shape[0] * 2 // 3:
                #print('best focus not within reasonable range')
                continue
            # take the image with focal distance equal to dist
            this_distance = np.where(focus_dists == dist)[0]
            if len(this_distance) == 0:
                continue
            this_distance = this_distance[0]
            #print(f'this distance is {this_distance}')
            image = this_patch_raw[this_distance]
            image = image.astype(np.float32)
            # bin by summing 2x2 pixels
            image = image.reshape(image.shape[0] // 2, 2, image.shape[1] // 2, 2).sum(axis=(1, 3))
            # check for nans and inf
            if np.isnan(image).any() or np.isinf(image).any():
                print('nan or inf')
                continue
            image = process_image(image, sat_prctile=95)
            tif_name = f'{subdir}/{file_identifier}_dist{dist}_patch{i}.tif'
            file_names.append(os.path.basename(tif_name))
            focal_distances.append(dist)
            # use PIL to save tif
            # image[image > 512] = 512
            im = Image.fromarray(image)
            im.save(tif_name)
            

    focal_distances = np.array(focal_distances)
    return file_names, focal_distances

# Parameters
patch_size = 250
threshold = 25  # Adjust the threshold based on your requirement

base_path = '/home/asaph/temp_data'

# find all subfolders that contains the string 'power20'
subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]# and 'power20' in f.name]
#subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]
#subfolders = subfolders[:2]

# loop over subfolders
all_file_names = []
all_distances = []
for subfolder in subfolders:
    mean_file_name = f'{subfolder}/mean_stack_{os.path.basename(subfolder)}.mat'
    single_file_name = f'{subfolder}/raw_stack_{os.path.basename(subfolder)}.mat'
    # Extract patches and distances
    file_names, distances = extract_patches(mean_file_name, single_file_name, patch_size, 20, threshold)
    all_file_names += file_names
    all_distances.append(distances)

all_distances = np.concatenate(all_distances)

# save file names and distances to the same csv file
with open(f'{subdir}/file_names_and_distances.csv', 'w') as f:
    f.write('file_name,distance\n')
    for file_name, distance in zip(all_file_names, all_distances):
        f.write(f'{file_name},{distance}\n')

