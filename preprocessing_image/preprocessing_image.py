import cv2
import numpy as np

def deskew(img, max_skew=10):

    # lấy chiều dài và chiều rộng 
    height, width = img.shape[:2]

    # chuyển thành ảnh Gray 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Khử nhiễu ảnh
    img_denoise = cv2.fastNlMeansDenoising(img_gray, h=3)

    # chuyen anh thanh anh nhi phan
    im_bw = cv2.threshold(img_denoise, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    #tìm đường thẳng trong ảnh nhị phân
    lines = cv2.HoughLinesP(
        im_bw, 2, np.pi / 180, 200, minLineLength=width / 12, maxLineGap=width / 150
    )

    if lines is None:
        return img  # Không tìm thấy đường thẳng nào trong ảnh, trả về ảnh gốc

    im_with_lines = img.copy() 
    for line in lines: 
        x1, y1, x2, y2 = line[0] 
        cv2.line(im_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2) # Vẽ đường thẳng màu xanh lá cây 
    # plt.imshow(cv2.cvtColor(im_with_lines, cv2.COLOR_BGR2RGB)) 
    # plt.title("Image with Detected Lines") 
    # plt.show()

    # Thu thập các góc của các đường thẳng này (radians)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angles.append(np.arctan2(y2 - y1, x2 - x1))
    # print("=========================",angles)

   # Kiểm tra xem phần lớn các đường thẳng có phải là dọc không
    landscape = np.sum([abs(angle) > np.pi / 4 for angle in angles]) > len(angles) / 2
    # print(landscape)


    # Lọc các góc để loại bỏ các giá trị ngoại lệ dựa trên max_skew
    if landscape:
        angles = [
            angle
            for angle in angles
            if np.deg2rad(90 - max_skew) < abs(angle) < np.deg2rad(90 + max_skew)
        ]
    else:
        angles = [angle for angle in angles if abs(angle) < np.deg2rad(max_skew)]
        # print(angles)
    # Không có dữ liệu lệch, trả về ảnh gốcgốc
    if len(angles) < 5:
        return img

    # Trung bình các góc để tính góc độ
    angle_deg = np.rad2deg(np.median(angles))
    # print(angle_deg)
    
    # Nếu đây là ảnh phong cảnh, xoay toàn bộ khung hình phù hợp
    if landscape:
        if angle_deg < 0:
            im = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            angle_deg += 90
        elif angle_deg > 0:
            im = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            angle_deg -= 90

    # Xoay ảnh theo góc độ còn lại
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1)
    im = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return im