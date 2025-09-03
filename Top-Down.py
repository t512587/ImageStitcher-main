import cv2
import numpy as np

# 讀取影像
img = cv2.imread("img/3.jpg")
clone = img.copy()

# 存放使用者點選的點
points = []

def mouse_click(event, x, y, flags, param):
    global points, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        # 左鍵點選角點
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")
        # 在影像上畫圈圈標記
        cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select 4 corners", clone)

        # 當點滿四個點時，自動計算
        if len(points) == 4:
            do_perspective_transform()

def do_perspective_transform():
    global points, img

    # step1: 取得四個點 (順序: 左上, 右上, 右下, 左下)
    src_points = np.array(points, dtype=np.float32)

    # step2: 定義轉換後的大小 (可依需求調整)
    W, H = 1920, 1080
    dst_points = np.array([
        [0, 0],      # 左上
        [W, 0],      # 右上
        [W, H],      # 右下
        [0, H]       # 左下
    ], dtype=np.float32)

    # step3: 計算透視變換矩陣
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # step4: warp 成鳥瞰圖
    top_down = cv2.warpPerspective(img, M, (W, H))

    cv2.imshow("Top-Down View", top_down)
    cv2.imwrite("output_topdown2.jpg", top_down)  # 可選：存檔
    print("✅ 透視變換完成，結果已顯示並存檔為 output_topdown.jpg")
cv2.namedWindow("Select 4 corners", cv2.WINDOW_NORMAL)  # 建立可調整大小的視窗
cv2.resizeWindow("Select 4 corners", 1280, 720)         # 設定顯示大小
cv2.imshow("Select 4 corners", clone)
cv2.setMouseCallback("Select 4 corners", mouse_click)
cv2.waitKey(0)
cv2.destroyAllWindows()
