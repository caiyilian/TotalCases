import mediapipe as mp
import cv2
import numpy as np

# 导入帽子图片
hat = cv2.imread("hat.jpeg")
# 把图片灰度化，用于把帽子和北京分割开
gray = cv2.cvtColor(hat, cv2.COLOR_BGR2GRAY)
# 获取帽子图片的高度和宽度
x_hat, y_hat, _ = hat.shape
# for循环，设置阈值，如果灰度图中某一点的像素值大于阈值，就把这个像素值变为0，只要阈值设置好，就可以把帽子和图片分离开
for i in range(x_hat):
    for j in range(y_hat):
        if gray[i, j] > 230:
            hat[i, j, :] = 0
# 初始化人脸检测器
mpFaceDetection = mp.solutions.face_detection  # 人脸识别

# 设置人脸检测的自信值，这个值设置得太高，可能会检测漏人脸，如果太低，可能会把不是人脸的检测成人脸
faceDetection = mpFaceDetection.FaceDetection(0.5)
# 初始化摄像头，0就是笔记本自带的摄像头参数
cap = cv2.VideoCapture(0)

while True:
    # 从摄像头中获取图片，如果获取到图片，success就是True，不过一般都是True，img就是获取到的图片
    success, img = cap.read()
    # 因为opencv里面的图片都是BGR也就是蓝绿红，但是一般图片都是红绿蓝RGB的，所以要把BGR格式转成RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 把图片放到人脸检测器去检测人脸
    results = faceDetection.process(imgRGB)
    # 定义一个空列表，用来装人脸信息，因为可能有多个人脸
    bboxs = []
    # 如果检测到人脸，进入if
    if results.detections:
        # for循环，循环每一张人脸
        for id, detection in enumerate(results.detections):
            # 人脸信息，里面包括人脸左上角的x，y值，以及这张人脸的高度宽度h和w，不过它进行了归一化操作，需要进一步处理
            bboxC = detection.location_data.relative_bounding_box
            # 先拿到真实图片的尺寸
            ih, iw, ic = img.shape
            # 把归一化的数据回复回来,得到人脸的真实位置
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            # 把这张人脸信息放到列表里面
            bboxs.append([id, bbox, detection.score])
            # 人脸的左上角点的x,y和宽度高度
            x, y, w, h = bbox
            #把帽子放到人脸上面，思路是帽子的宽度和人脸的宽度相同
            try:
                # 帽子的理论高度，把帽子的宽度变化之后，如果要保证长宽比不变，高度理论上会变为多少
                au_h = w * x_hat // y_hat
                # 帽子实际高度，因为如果你人脸里图片顶部太近，可能帽子会超出图片，这个时候，就让帽子的图片高度是你人脸距离图片顶部的高度
                hat_height = au_h if y > au_h else y
                # 把帽子的宽度变为人脸的宽度，高度变为上面求出来的实际高度
                hat_resize = cv2.resize(hat, (w, hat_height))
                # 掩膜，用处是，设置哪些地方需要用原始图片的像素，哪些地方用帽子的像素
                mask = (hat_resize == 0).astype(np.uint8)
                # 利用掩膜，把帽子精准的嵌入到人脸上面
                img[y - hat_height:y, x:x + w, :] = img[y - hat_height:y, x:x + w, :] * mask + hat_resize
            # 人脸靠近左边界或者右边界的时候，会报错，捕获这个错误，以防止程序停止
            except:
                pass
            # 展示图片
            cv2.imshow('image', img)
            # 图片刷新率，每隔多少ms刷新一次图片
            cv2.waitKey(1)
