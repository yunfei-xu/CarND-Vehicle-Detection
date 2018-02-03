import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
from skimage.feature import hog
import glob
from scipy.ndimage.measurements import label


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


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


def find_cars(img, ystart, ystop, scale, svc, X_scaler,
              orient, pix_per_cell, cell_per_block,
              spatial_size=0, hist_bins=0):
    """
    Define a single function that can extract features using hog sub-sampling and make predictions.
    """
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    bboxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            # spatial_features = bin_spatial(subimg, size=spatial_size)
            # hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            # test_features = X_scaler.transform(
            #     np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_features = X_scaler.transform(hog_features.reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bbox = ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart))
                bboxes.append(bbox)

    return bboxes


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Define a function to draw bounding boxes
    """
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


with open('data.p', 'rb') as f:
    data = pickle.load(f)
    svc = data["svc"]
    X_scaler = data["X_scaler"]
    orient = data["orient"]
    pix_per_cell = data["pix_per_cell"]
    cell_per_block = data["cell_per_block"]

ystart = 400
ystop = 656
scales = [1.0, 1.5, 2.0]

# Show the raw detected bounding boxes for the test images
image_files = glob.glob('./test_images/*.jpg')
fig = plt.figure(figsize=(10, 10))
for i, image_file in enumerate(image_files):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = []
    for scale in scales:
        bboxes += find_cars(image, ystart, ystop, scale, svc, X_scaler, orient,
                            pix_per_cell, cell_per_block, spatial_size=0, hist_bins=0)
    out_img = draw_boxes(image, bboxes, color=(0, 0, 255), thick=6)
    ax = plt.subplot(3, 2, i + 1)
    ax.imshow(out_img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(image_file.split('/')[-1])
plt.tight_layout()
plt.savefig('./output_images/test_images_detection.png', bbox_inches='tight')

# Show the detection result on 5 sequencial video frames extracted from test_video.mp4
# image_files = glob.glob('./test_video_frames/*.png')
image_files = glob.glob('./project_video_frames/*.png')
start_frame = 0
n_frames = 5
fig, axes = plt.subplots(n_frames, 2, figsize=(8, 10))
for i in range(n_frames):
    image = cv2.imread(image_files[i + start_frame])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = []
    for scale in scales:
        bboxes += find_cars(image, ystart, ystop, scale, svc, X_scaler, orient,
                            pix_per_cell, cell_per_block, spatial_size=0, hist_bins=0)
    out_img = draw_boxes(image, bboxes, color=(0, 0, 255), thick=6)
    axes[i, 0].imshow(out_img)
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = add_heat(heat, bboxes)
    heatmap = np.clip(heat, 0, 255)
    axes[i, 1].imshow(heatmap, cmap='hot')
    axes[i, 1].set_xticks([])
    axes[i, 1].set_yticks([])
plt.tight_layout()
plt.savefig('./output_images/single_frame_detection.png', bbox_inches='tight')

# Show the final result of the above 5 frames
bboxes = []
image = None
for i in range(n_frames):
    image = cv2.imread(image_files[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for scale in scales:
        bboxes += find_cars(image, ystart, ystop, scale, svc, X_scaler, orient,
                            pix_per_cell, cell_per_block, spatial_size=0, hist_bins=0)
out_img = draw_boxes(image, bboxes, color=(0, 0, 255), thick=6)
# image = cv2.imread(image_files[n_frames - 1])
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
heat = np.zeros_like(image[:, :, 0]).astype(np.float)
# Add heat to each box in box list
heat = add_heat(heat, bboxes)
# Apply threshold to help remove false positives
heat = apply_threshold(heat, 3)
# Visualize the heatmap when displaying
heatmap = np.clip(heat, 0, 255)
# Find final boxes from heatmap using label function
labels = label(heatmap)
print('{} cars found'.format(labels[1]))
draw_img = draw_labeled_bboxes(np.copy(image), labels)


fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].imshow(draw_img)
axes[0, 0].set_xticks([])
axes[0, 0].set_yticks([])
axes[0, 0].set_title('Final result')
axes[0, 1].imshow(out_img)
axes[0, 1].set_xticks([])
axes[0, 1].set_yticks([])
axes[0, 1].set_title('Combined bboxes')
axes[1, 0].imshow(heatmap, cmap='hot')
axes[1, 0].set_xticks([])
axes[1, 0].set_yticks([])
axes[1, 0].set_title('Combined heatmap')
axes[1, 1].imshow(labels[0], cmap='gray')
axes[1, 1].set_xticks([])
axes[1, 1].set_yticks([])
axes[1, 1].set_title('Labels')
plt.tight_layout()
plt.savefig('./output_images/final_result.png', bbox_inches='tight')
