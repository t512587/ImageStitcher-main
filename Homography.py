import numpy as np
import cv2

class Homography:
    def solve_homography(self, P, m):
        """
        Solve homography matrix

        Args:
            P:  Coordinates of the points in the original plane,
            m:  Coordinates of the points in the target plane


        Returns:
            H: Homography matrix
        """
        try:
            A = []
            for r in range(len(P)):
                # print(m[r, 0])
                A.append([-P[r, 0], -P[r, 1], -1, 0, 0, 0, P[r, 0] * m[r, 0], P[r, 1] * m[r, 0], m[r, 0]])
                A.append([0, 0, 0, -P[r, 0], -P[r, 1], -1, P[r, 0] * m[r, 1], P[r, 1] * m[r, 1], m[r, 1]])

            u, s, vt = np.linalg.svd(A)  # Solve s ystem of linear equations Ah = 0 using SVD
            # pick H from last line of vt
            H = np.reshape(vt[8], (3, 3))
            # normalization, let H[2,2] equals to 1
            H = (1 / H.item(8)) * H
        except:
            print("Error occur!")

        return H