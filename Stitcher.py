import random
import numpy as np
import cv2
from matplotlib import pyplot as plt


class Stitcher:
    def __init__(self):
        pass

    def stitch(self, imgs, blending_mode="linearBlending", ratio=0.75):
        '''
            The main method to stitch image
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        print("Left img size (", hl, "*", wl, ")")
        print("Right img size (", hr, "*", wr, ")")

        # Step1 - extract the keypoints and features by SIFT detector and descriptor
        print("Step1 - Extract the keypoints and features by SIFT detector and descriptor...")
        kps_l, features_l = self.detectAndDescribe(img_left)
        kps_r, features_r = self.detectAndDescribe(img_right)

        # Step2 - extract the match point with threshold (David Lowe’s ratio test)
        print("Step2 - Extract the match point with threshold (David Lowe’s ratio test)...")
        matches_pos = self.matchKeyPoint(kps_l, kps_r, features_l, features_r, ratio)
        print("The number of matching points:", len(matches_pos))

        # Step2 - draw the img with matching point and their connection line
        self.drawMatches([img_left, img_right], matches_pos)

        # Step3 - fit the homography model with RANSAC algorithm
        print("Step3 - Fit the best homography model with RANSAC algorithm...")
        HomoMat = self.fitHomoMat(matches_pos)
        print(HomoMat)
        # Step4 - Warp image to create panoramic image
        print("Step4 - Warp image to create panoramic image...")
        warp_img = self.warp([img_left, img_right], HomoMat, blending_mode)

        return warp_img

    def detectAndDescribe(self, img):
        '''
        The Detector and Descriptor
        '''
        sift = cv2.SIFT_create()
        kps, features = sift.detectAndCompute(img, None)
        return kps, features

    def matchKeyPoint(self, kps_l, kps_r, features_l, features_r, ratio=0.75):
        """
        Match keypoints between two images using OpenCV BFMatcher with ratio test.
        Returns list of matched points: [[(x1, y1), (x2, y2)], ...]
        """
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(features_l, features_r, k=2)

        goodMatches_pos = []
        for m, n in matches:
            if m.distance < ratio * n.distance:  # Lowe ratio test
                pt_left = kps_l[m.queryIdx].pt
                pt_right = kps_r[m.trainIdx].pt
                goodMatches_pos.append([(int(pt_left[0]), int(pt_left[1])),
                                        (int(pt_right[0]), int(pt_right[1]))])
        return goodMatches_pos

    def drawMatches(self, imgs, matches_pos):
        '''
            Draw the match points img with keypoints and connection line
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        vis = np.zeros((max(hl, hr), wl + wr, 3), dtype="uint8")
        vis[0:hl, 0:wl] = img_left
        vis[0:hr, wl:] = img_right

        for (img_left_pos, img_right_pos) in matches_pos:
            pos_l = img_left_pos
            pos_r = img_right_pos[0] + wl, img_right_pos[1]
            cv2.circle(vis, pos_l, 3, (0, 0, 255), 1)
            cv2.circle(vis, pos_r, 3, (0, 255, 0), 1)
            cv2.line(vis, pos_l, pos_r, (255, 0, 0), 1)

        plt.figure(4)
        plt.title("img with matching points")
        plt.imshow(vis[:, :, ::-1])

        return vis

    def fitHomoMat(self, matches_pos):
        '''
            Fit the best homography model with RANSAC algorithm
        '''
        dstPoints = np.array([list(dst) for dst, src in matches_pos], dtype=np.float32)
        srcPoints = np.array([list(src) for dst, src in matches_pos], dtype=np.float32)

        H, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, ransacReprojThreshold=5.0)
        num_inliers = np.sum(mask)
        print("The Number of Maximum Inlier:", num_inliers)

        return H

    def warp(self, imgs, HomoMat, blending_mode):
        '''
           Warp image to create panoramic image
        '''
        img_left, img_right = imgs
        hl, wl = img_left.shape[:2]
        hr, wr = img_right.shape[:2]

        # 輸出影像大小：寬 = wl + wr，高 = max(hl, hr)
        stitch_width = wl + wr
        stitch_height = max(hl, hr)

        # warp 右圖
        warped_right = cv2.warpPerspective(
            img_right,
            HomoMat,
            (stitch_width, stitch_height),
            flags=cv2.INTER_LINEAR
        )

        # 初始化拼接圖
        stitch_img = np.zeros((stitch_height, stitch_width, 3), dtype=np.uint8)
        stitch_img[:hl, :wl] = img_left

        if blending_mode == "noBlending":
            mask = (warped_right > 0)
            stitch_img[mask] = warped_right[mask]

        # 移除黑邊 (快速版)
        stitch_img = self.removeBlackBorder(stitch_img)
        return stitch_img

    def removeBlackBorder(self, img):
        '''
        Fast remove black border using cv2.findNonZero
        '''
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
        if coords is None:  # 全黑避免錯誤
            return img
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]
