import cv2
import numpy as np


class ImageStitcher:
    def __init__(self):
        # 預定義的透視變換點座標
        # cam0 座標點 (左上, 右上, 右下, 左下)
        self.cam0_src_points = np.array([
            [0, 180],
            [1080, 180], 
            [1080, 1990],
            [0, 1500]
        ], dtype=np.float32)
        
        # cam1 座標點 (左上, 右上, 右下, 左下)
        self.cam1_src_points = np.array([
            [0, 256],
            [1080, 256],
            [1080, 1880], 
            [0, 1880]
        ], dtype=np.float32)
        
        # 目標輸出大小
        self.output_width = 1920
        self.output_height = 1080
        
        # 目標點 (標準矩形)
        self.dst_points = np.array([
            [0, 0],                                    # 左上
            [self.output_width, 0],                    # 右上
            [self.output_width, self.output_height],   # 右下
            [0, self.output_height]                    # 左下
        ], dtype=np.float32)
        
        # 計算透視變換矩陣
        self.cam0_transform_matrix = cv2.getPerspectiveTransform(
            self.cam0_src_points, self.dst_points
        )
        self.cam1_transform_matrix = cv2.getPerspectiveTransform(
            self.cam1_src_points, self.dst_points
        )
        
        print("✅ 透視變換矩陣已計算完成")
        print(f"Cam0 變換矩陣:\n{self.cam0_transform_matrix}")
        print(f"Cam1 變換矩陣:\n{self.cam1_transform_matrix}")

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

    def apply_perspective_transform(self, img, camera_id):
        """對指定攝影機影像進行透視變換"""
        if camera_id == 0:
            transform_matrix = self.cam0_transform_matrix
        elif camera_id == 1:
            transform_matrix = self.cam1_transform_matrix
        else:
            raise ValueError("camera_id 必須是 0 或 1")
            
        # 執行透視變換
        warped = cv2.warpPerspective(
            img, 
            transform_matrix, 
            (self.output_width, self.output_height),
            flags=cv2.INTER_LINEAR
        )
        return warped

    def warp_and_stitch(self, imgs, HomoMat):
        """先進行透視變換，再進行拼接"""
        img_left, img_right = imgs
        
        # 分別對兩個攝影機影像進行透視變換
        warped_left = self.apply_perspective_transform(img_left, 0)
        warped_right = self.apply_perspective_transform(img_right, 1)
        
        # 現在使用變換後的影像進行拼接
        hl, wl = warped_left.shape[:2]
        hr, wr = warped_right.shape[:2]

        # 輸出影像大小調大，避免右圖被截斷
        stitch_width = wl + wr
        stitch_height = max(hl, hr)

        # 對右圖進行 Homography 變換
        final_warped_right = cv2.warpPerspective(
            warped_right,
            HomoMat,
            (stitch_width, stitch_height),
            flags=cv2.INTER_LINEAR
        )

        # 初始化拼接圖
        stitch_img = np.zeros((stitch_height, stitch_width, 3), dtype=np.uint8)
        stitch_img[:hl, :wl] = warped_left

        # 疊加右圖
        mask = (final_warped_right > 0)
        stitch_img[mask] = final_warped_right[mask]
        stitch_img = self.removeBlackBorder(stitch_img)
        return stitch_img


if __name__ == "__main__":
    # 左右相機 index
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(1)

    # 設定兩台攝影機解析度為 1920x1080
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # 固定 Homography (用於拼接已透視變換的影像)

    HomoMat = np.array([
        [1.16145886e-01,  1.37416633e-02 ,   1.63941242e+03],
        [-2.03193604e-01,  9.19190469e-01,  2.39993756e+01],
        [-3.35329508e-04,  4.64436208e-05,  1.00000000e+00]
    ])

    stitcher = ImageStitcher()

    while True:
        ret_left, img_left = cap_left.read()
        ret_right, img_right = cap_right.read()

        if not ret_left or not ret_right:
            print("⚠️ 相機影像讀取失敗")
            break

        # 旋轉 (如果需要的話)
        img_left = cv2.rotate(img_left, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 左轉90
        img_right = cv2.rotate(img_right, cv2.ROTATE_90_CLOCKWISE)       # 右轉90

        # 先進行透視變換，再拼接
        stitched = stitcher.warp_and_stitch((img_left, img_right), HomoMat)

        # 縮小拼接結果以適合螢幕顯示
        stitch_height, stitch_width = stitched.shape[:2]
        display_width = 1280  # 顯示寬度
        display_height = int(stitch_height * display_width / stitch_width)
        
        # 如果高度還是太大，進一步縮小
        if display_height > 720:
            display_height = 720
            display_width = int(stitch_width * display_height / stitch_height)
            
        stitched_display = cv2.resize(stitched, (display_width, display_height))
        
        # 顯示縮小後的結果
        cv2.imshow("Panorama", stitched_display)
        
        # 可選：顯示個別透視變換結果（也縮小顯示）
        warped_left = stitcher.apply_perspective_transform(img_left, 0)
        warped_right = stitcher.apply_perspective_transform(img_right, 1)
        
        warped_left_small = cv2.resize(warped_left, (640, 360))
        warped_right_small = cv2.resize(warped_right, (640, 360))
        
        cv2.imshow("Warped Left", warped_left_small)
        cv2.imshow("Warped Right", warped_right_small)

        # 按 q 離開
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()




