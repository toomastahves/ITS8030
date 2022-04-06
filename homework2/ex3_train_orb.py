import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def train():
    train_keypoints = []
    train_descriptors = []

    # Iterate over train images
    for i in range(6,36):
        print('Training image {}'.format(i))
        query_image = 'lusitania_{}.png'.format(i)
        train_image = 'lusitania_{}.png'.format(i)
        img1 = cv2.imread('images/templates/{}'.format(query_image), 0)
        img2 = cv2.imread('images/original/{}'.format(train_image), 0)

        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)

        # Hack to initialize array with correct shape
        if(i == 6):
            train_descriptors = des1

        train_keypoints = np.concatenate((train_keypoints, kp1), axis=0)
        train_descriptors = np.concatenate((train_descriptors, des1), axis=0)
    
    print('Training finished')
    return train_descriptors, train_keypoints

def test(train_descriptors, train_keypoints):
    # Iterate over test images
    for test_image_id in range(1,6):
        print('Testing image {}'.format(test_image_id))
        img3 = cv2.imread('images/test/lusitania_{}.png'.format(test_image_id), 0)

        # Use ORB to find descriptor on test image
        orb = cv2.ORB_create()
        kp3, des3 = orb.detectAndCompute(img3, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(train_descriptors, des3)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # Get indexes of good points for drawing
        filter_kp_idx = []
        for p in matches:
            filter_kp_idx.append(p.trainIdx)
        
        # Get keypoint locations on test image
        good_kp = np.array(kp3)[filter_kp_idx]
        print('Found {} matching points'.format(len(good_kp)))

        # Convert tuple to array
        rectangle_pts = []
        for p in good_kp:
            rectangle_pts.append(p.pt)
        rectangle_pts = np.asarray(rectangle_pts)

        # Remove outliers using Gaussian with sigma = 1.5
        rectangle_pts_without_outliers = remove_outliers(rectangle_pts, 1.5)

        # Calculate centroid
        centroid = find_centroid(rectangle_pts_without_outliers)

        # Find min and max for rectangle
        min_x, min_y = np.int32(rectangle_pts_without_outliers.min(axis=0))
        max_x, max_y = np.int32(rectangle_pts_without_outliers.max(axis=0))

        # Draw good matching keypoints on image and save
        img4 = cv2.drawKeypoints(img3, good_kp, None,color=(0,255,0), flags=None)
        cv2.rectangle(img4,(min_x, min_y), (max_x,max_y), color=(255,0,0), thickness=3)
        cv2.circle(img4, centroid, 5, (0, 0, 255), 3)
        cv2.imwrite('ex3_result/ex3_result_lusitania_orb_{}.png'.format(test_image_id), img4)

def find_centroid(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return (np.int32(sum_x/length), np.int32(sum_y/length))

def remove_outliers(values, sigma):
    xpoints = values[:,0]
    ypoints = values[:,1]
    xmean = np.mean(xpoints)
    ymean = np.mean(ypoints)
    xstandard_deviation = np.std(xpoints)
    ystandard_deviation = np.std(ypoints)
    xdistance_from_mean = abs(xpoints - xmean)
    ydistance_from_mean = abs(ypoints - ymean)
    xnot_outlier = xdistance_from_mean < sigma * xstandard_deviation
    ynot_outlier = ydistance_from_mean < sigma * ystandard_deviation
    combined = np.logical_and(np.array(xnot_outlier), np.array(ynot_outlier))
    no_outliers = values[combined]
    return no_outliers


if __name__ == '__main__':
    train_descriptors, train_keypoints = train()
    test(train_descriptors, train_keypoints)
