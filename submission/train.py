import numpy as np
import glob
import time
import cv2
import random
import re
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage import exposure
from sklearn.model_selection import train_test_split


DEBUG = True


def convert_color(image, cspace='RGB'):
    if colorspace != 'RGB':
        if colorspace == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif colorspace == 'LUV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif colorspace == 'HLS':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif colorspace == 'YUV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif colorspace == 'YCrCb':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    return image


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    """
    Define a function to return HOG features and visualization
    """
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def extract_features(imgs, cspace='RGB', orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0):
    """
    Define a function to extract features from a list of images
    Have this function call bin_spatial() and color_hist()
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features

# Divide up into cars and notcars
cars = glob.glob('./dataset/vehicles/*/*.png')
notcars = glob.glob('./dataset/non-vehicles/*/*.png')
print('Dataset contains {} vehicle images'.format(len(cars)))
print('Dataset contains {} non-vehicle images'.format(len(notcars)))

# Experiment with features
colorspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
colorspace_cap = re.findall('[A-Z][^A-Z]*', colorspace)
print(colorspace_cap)
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"

if DEBUG:
    # Show some car images
    fig = plt.figure(figsize=(6, 6))
    car_samples = random.sample(cars, 25)
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(mpimg.imread(car_samples[i]))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('./output_images/car_samples.png', bbox_inches='tight')

    # Show some notcar images
    fig = plt.figure(figsize=(6, 6))
    notcar_samples = random.sample(notcars, 25)
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(mpimg.imread(notcar_samples[i]))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('./output_images/notcar_samples.png', bbox_inches='tight')

    # Show hog features applied to some car images from training set
    fig, axes = plt.subplots(4, 7, figsize=(10, 8))
    for i in range(4):
        image = mpimg.imread(car_samples[i])
        axes[i, 0].imshow(image)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_title('Input image')
        image = convert_color(image, cspace=colorspace)
        for j in range(image.shape[-1]):
            features, hog_image = hog(image[:, :, j],
                                      orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      transform_sqrt=True,
                                      visualise=True,
                                      feature_vector=True)
            # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            axes[i, 2 * j + 1].imshow(image[:, :, j], cmap='gray')
            axes[i, 2 * j + 1].set_xticks([])
            axes[i, 2 * j + 1].set_yticks([])
            axes[i, 2 * j + 1].set_title(colorspace_cap[j])
            axes[i, 2 * j + 2].imshow(hog_image, cmap='gray')
            axes[i, 2 * j + 2].set_xticks([])
            axes[i, 2 * j + 2].set_yticks([])
            axes[i, 2 * j + 2].set_title(colorspace_cap[j] + '-HOG')
    plt.savefig('./output_images/car_hog_images.png', bbox_inches='tight')

    # Show hog features applied to some non-car images from training set
    fig, axes = plt.subplots(4, 7, figsize=(10, 8))
    for i in range(4):
        image = mpimg.imread(notcar_samples[i])
        axes[i, 0].imshow(image)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_title('Input image')
        image = convert_color(image, cspace=colorspace)
        for j in range(image.shape[-1]):
            features, hog_image = hog(image[:, :, j],
                                      orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      transform_sqrt=True,
                                      visualise=True,
                                      feature_vector=True)
            # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            axes[i, 2 * j + 1].imshow(image[:, :, j], cmap='gray')
            axes[i, 2 * j + 1].set_xticks([])
            axes[i, 2 * j + 1].set_yticks([])
            axes[i, 2 * j + 1].set_title(colorspace_cap[j])
            axes[i, 2 * j + 2].imshow(hog_image, cmap='gray')
            axes[i, 2 * j + 2].set_xticks([])
            axes[i, 2 * j + 2].set_yticks([])
            axes[i, 2 * j + 2].set_title(colorspace_cap[j] + '-HOG')
    plt.savefig('./output_images/notcar_hog_images.png', bbox_inches='tight')

# Extract features
print('Extracting features...')
t1 = time.time()
car_features = extract_features(cars, cspace=colorspace, orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                hog_channel=hog_channel)
notcar_features = extract_features(notcars, cspace=colorspace, orient=orient,
                                   pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                   hog_channel=hog_channel)
t2 = time.time()
print('{:.2f} seconds to extract HOG features...'.format(t2 - t1))
print('Feature size: {}'.format(car_features[0].shape))

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
# rand_state = np.random.randint(0, 100)
rand_state = 1
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using: {} orientations, {} pixels per cell and {} cells per block'.format(orient, pix_per_cell, cell_per_block))
print('Feature vector length: {}'.format(len(X_train[0])))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t1 = time.time()
print('Training SVC classifier...')
svc.fit(X_train, y_train)
t2 = time.time()
print('{:.2f} seconds to train the classifier...'.format(t2 - t1))
# Check the score of the SVC
print('Test Accuracy of SVC = {:.4f}'.format(svc.score(X_test, y_test)))

# Save the result
data = {'svc': svc,
        'X_scaler': X_scaler,
        'orient': orient,
        'pix_per_cell': pix_per_cell,
        'cell_per_block': cell_per_block}
with open('data.p', 'wb') as f:
    pickle.dump(data, f)
