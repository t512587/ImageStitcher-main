import numpy as np
import matplotlib.pyplot as plt
import cv2  # cv2 version-> 3.4.2 (The version is important for using SIFT)
import random  # using for RANSAC algorithm
from Stitcher import Stitcher

if __name__ == "__main__":

    fileNameList = [('5', '6')]
    for fname1, fname2 in fileNameList:
        # Read the img file
        src_path = "img/"
        fileName1 = fname1
        fileName2 = fname2
        img_left = cv2.imread(src_path + fileName1 + ".jpg")
        img_right = cv2.imread(src_path + fileName2 + ".jpg")

        # The stitch object to stitch the image
        blending_mode = "noBlending"  # three mode - noBlending、linearBlending、linearBlendingWithConstant
        stitcher = Stitcher()
        stitch_img = stitcher.stitch([img_left, img_right], blending_mode)

        # plot the stitched image
        plt.figure(0)
        plt.title("stitch_img")
        plt.imshow(stitch_img[:, :, ::-1].astype(int))
        plt.axis("off")
    
    # Show all figures after processing
    plt.show()