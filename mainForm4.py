from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QLabel
from PIL import Image
import pyzbar.pyzbar as pyzbar
from GUI4 import *
import cv2
import matplotlib.pyplot as plt
import imutils
import os
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"E:/Yan.All2/Tesseract-OCR/tesseract.exe"
import logging
from paddleocr import PaddleOCR
logging.getLogger('paddleocr').setLevel(logging.WARNING)
# 初始化车牌检测器
carplate_haar = cv2.CascadeClassifier(
    r"E:\Tool\GitHub\Opencv-master\opencv-master\opencv-master\data\haarcascades\haarcascade_russian_plate_number.xml")
ocr = PaddleOCR(use_angle_cls=True)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        cv2.ocl.setUseOpenCL(False)
        self.pushButton.clicked.connect(self.QRimage)  # 连接按钮的 clicked 信号到 QRimage 方法
        self.pushButton_2.clicked.connect(self.QRvideo)  # 连接按钮的 clicked 信号到 QRvideo 方法
        self.pushButton_3.clicked.connect(self.QRcamera) # 连接摄像头识别二维码按钮的信号到 QRcamera 方法
        self.pushButton_4.clicked.connect(self.IDimage)
        self.pushButton_5.clicked.connect(self.IDvideo)
        self.pushButton_6.clicked.connect(self.IDcamera)
        self.pushButton_7.clicked.connect(self.faceimage)
        self.pushButton_8.clicked.connect(self.facevideo)
        self.pushButton_9.clicked.connect(self.facecamera)
        self.pushButton_10.clicked.connect(self.choiceA)
        self.pushButton_11.clicked.connect(self.choiceB)
        self.pushButton_12.clicked.connect(self.AandB)
        self.imageA = None  # 初始化图片A
        self.imageB = None  # 初始化图片B

    def choiceB(self):
        # 用户选择图片B
        self.select_image('B')

    def choiceA(self):
        # 用户选择图片A
        self.select_image('A')

    def select_image(self, image_type):
        # 弹出文件选择对话框，选择图片
        filename, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            # 读取图片
            img = cv2.imread(filename)
            if img is None:
                QMessageBox.warning(self, "警告", "无法加载图片，请选择有效的图片文件。")
            else:
                # 显示图片
                cv2.imshow(f"Selected {image_type} Image", img)
                cv2.waitKey(0)  # 等待用户按下任意键以关闭窗口
                # 根据选择的图片类型存储图片
                if image_type == 'A':
                    self.imageA = img
                elif image_type == 'B':
                    self.imageB = img
                else:
                    QMessageBox.warning(self, "警告", "无效的图片类型。")


    def AandB(self):
        img_left = self.imageA
        img_right = self.imageB
        # 模块一：提取特征
        kps_left, features_left = MainWindow.detect(img_left)
        kps_right, features_right = MainWindow.detect(img_right)
        # 模块二：特征匹配
        matches, H, good = MainWindow.match_keypoints(kps_left, kps_right, features_left, features_right, 0.5, 0.99)
        # 模块三：透视变换-拼接
        vis = MainWindow.drawMatches(img_left, img_right, kps_left, kps_right, matches, H)
        # 显示拼接图形
        # plt.figure(),plt.axis('off')
        # plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        # plt.show()
        qt_image = self.convert_cv_qt(vis)
        # 显示在 label_4 中
        self.label_4.setPixmap(QPixmap.fromImage(qt_image))

    def convert_cv_qt(self, cv_img):

        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(981, 581, Qt.KeepAspectRatio)
        return p



    def detect(image):
        # 创建SIFT生成器
        # descriptor是一个对象，这里使用的是SIFT算法
        descriptor = cv2.SIFT_create()
        # 检测特征点及其描述子（128维向量）
        kps, features = descriptor.detectAndCompute(image, None)
        return (kps, features)


    def match_keypoints(kps_left, kps_right, features_left, features_right, ratio, threshold):
        # kps_left,kps_right,features_left,features_right: 两幅图形的特征点坐标及特征向量
        # 创建暴力匹配器
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        # 使用knn检测，匹配left,right图的特征点
        raw_matches = matcher.knnMatch(features_left, features_right, 2)
        print('左右图的匹配特征点数：', len(raw_matches))
        matches = []  # 存坐标，为了后面
        good = []  # 存对象，为了后面的演示
        # 筛选匹配点
        for m in raw_matches:
            # 筛选条件
            #  print(m[0].distance,m[1].distance)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                good.append([m[0]])
                matches.append((m[0].queryIdx, m[0].trainIdx))
                """
                queryIdx：测试图像的特征点描述符的下标==>img_left
                trainIdx：样本图像的特征点描述符下标==>img_right
                distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
                """
        # 特征点对数大于4就够用来构建变换矩阵了
        kps_left = np.float32([kp.pt for kp in kps_left])
        kps_right = np.float32([kp.pt for kp in kps_right])
        print('筛选后匹配点数:', len(matches))
        if len(matches) > 4:
            # 获取匹配点坐标
            pts_left = np.float32([kps_left[i] for (i, _) in matches])
            pts_right = np.float32([kps_right[i] for (_, i) in matches])
            # 计算变换矩阵(采用ransac算法从pts中选择一部分点)
            H, status = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, threshold)
            return (matches, H, good)
        return None

    def drawMatches(img_left, img_right, kps_left, kps_right, matches, H):
        # 获取图片宽度和高度
        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]
        """对imgB进行透视变换
        由于透视变换会改变图片场景的大小，导致部分图片内容看不到
        所以对图片进行扩展:高度取最高的，宽度为两者相加"""
        image = np.zeros((max(h_left, h_right), w_left + w_right, 3), dtype='uint8')
        # 初始化
        image[0:h_left, 0:w_left] = img_right
        # 利用以获得的单应性矩阵进行变透视换"""
        image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))  # (w,h
        # 将透视变换后的图片与另一张图片进行拼接"""
        image[0:h_left, 0:w_left] = img_left
        return image

    def carplate_detect(image):#检测图像中的车牌区域。
        carplate_rects = carplate_haar.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3)
        return carplate_rects

    def carplate_extract(image, rects):#从原始图像中裁剪出车牌图像。
        plates = []
        for x, y, w, h in rects:
            carplate_img = image[y + 15:y + h - 10, x + 15:x + w - 20]
            plates.append(carplate_img)
        return plates

    def ocr_plate(image):#光学字符识别
        results = ocr.ocr(image, cls=True)
        if results is None or len(results) == 0:
            return "No plate detected"
        else:
            try:
                # 返回第一个识别结果
                return results[0][0][1][0]
            except IndexError:
                # 如果索引操作失败，返回未检测到车牌
                return "No plate detected"

    def IDimage(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        detected_plates = []
        if filename:
            # 读取图片
            img = cv2.imread(filename)
            cv2.imshow('Original Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            if img is None:
                QMessageBox.warning(self, "警告", "无法加载图片，请选择有效的图片文件。")
                return
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = MainWindow.carplate_detect(gray)
            for plate_img in MainWindow.carplate_extract(gray, rects):
                plate_str = MainWindow.ocr_plate(plate_img)
                if plate_str != "No plate detected":  # 过滤掉未检测到车牌的情况
                    detected_plates.append(plate_str)

            # 将列表转换为字符串，用逗号和空格分隔
            plates_str = ', '.join(detected_plates)

            # 设置标签的文本
            self.label_2.setText(plates_str)
            self.label_2.setAlignment(Qt.AlignCenter)

    def IDvideo(self):
        # 弹出选择文件对话框，选择视频文件
        filename, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if not filename:
            print("No video selected.")
            return

        # 初始化视频读取器
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            QMessageBox.warning(self, "警告", "无法打开视频文件，请选择有效的视频文件。")
            return

        # 清空标签的文本
        self.label_2.clear()

        tick = cv2.getTickCount()
        while True:
            # 读取视频的下一帧
            ret, frame = cap.read()
            if not ret:
                break  # 如果视频结束或无法读取帧，则退出循环

            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 车牌检测
            rects = MainWindow.carplate_detect(gray)  # 确保这里调用的是实例方法

            # 存储检测到的车牌号，同时去除重复
            unique_plates = set()

            for plate_img in MainWindow.carplate_extract(gray, rects):  # 同上
                plate_str = MainWindow.ocr_plate(plate_img)  # 同上
                if plate_str != "No plate detected":
                    unique_plates.add(plate_str)

            # 将集合转换为字符串，用逗号和空格分隔
            plates_str = ', '.join(unique_plates)

            # 更新UI，显示车牌号码
            self.label_2.setText(plates_str)
            self.label_2.setAlignment(Qt.AlignCenter)

            # 处理所有等待的事件，确保UI响应性
            QApplication.processEvents()

            # 按 'q' 退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            delay = max(0, 33 - (cv2.getTickCount() - tick), )
            cv2.waitKey(delay)  # 等待一定的毫秒数

            # 更新下一次循环的开始时间
            tick = cv2.getTickCount()
            cv2.imshow('Video', frame)



        # 释放视频读取器并关闭所有OpenCV窗口
        cap.release()
        cv2.destroyAllWindows()

        # 最后更新一次标签，以显示最后一次检测到的车牌号码
        self.label_2.setText(plates_str)
        self.label_2.setAlignment(Qt.AlignCenter)

    def IDcamera(self):
        # 初始化摄像头
        cap = cv2.VideoCapture(0)  # 参数0通常表示系统的默认摄像头

        if not cap.isOpened():
            QMessageBox.warning(self, "警告", "无法打开摄像头，请检查摄像头连接。")
            return

        # 清空标签的文本
        self.label_2.clear()

        try:
            while True:
                # 读取摄像头的下一帧
                ret, frame = cap.read()
                if not ret:
                    break  # 如果摄像头结束或无法读取帧，则退出循环

                # 转换为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 车牌检测
                rects = MainWindow.carplate_detect(gray)

                # 存储检测到的车牌号，同时去除重复
                unique_plates = set()

                for plate_img in MainWindow.carplate_extract(gray, rects):
                    plate_str = MainWindow.ocr_plate(plate_img)
                    if plate_str != "No plate detected":
                        unique_plates.add(plate_str)

                # 将集合转换为字符串，用逗号和空格分隔
                plates_str = ', '.join(unique_plates)

                # 更新UI，显示车牌号码
                self.label_2.setText(plates_str)
                self.label_2.setAlignment(Qt.AlignCenter)

                # 处理所有等待的事件，确保UI响应性
                QApplication.processEvents()

                # 按 'q' 退出循环
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # 释放摄像头并关闭所有OpenCV窗口
            cap.release()
            cv2.destroyAllWindows()

        # 最后更新一次标签，以显示最后一次检测到的车牌号码
        self.label_2.setText(plates_str)
        self.label_2.setAlignment(Qt.AlignCenter)

    def faceimage(self):
        # 弹出文件选择对话框，选择图片
        filename, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            # 读取图片
            img = cv2.imread(filename)
            if img is None:
                QMessageBox.warning(self, "警告", "无法加载图片，请选择有效的图片文件。")
                return

            # 将图片转换为灰度图，用于检测
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 加载 Haar 特征级联文件
            face_cascade = cv2.CascadeClassifier(
                "E:\Tool\GitHub\Opencv-master\opencv-master\opencv-master\data\haarcascades\haarcascade_frontalface_alt.xml")
            eye_cascade = cv2.CascadeClassifier(
                "E:\Tool\GitHub\Opencv-master\opencv-master\opencv-master\data\haarcascades\haarcascade_eye_tree_eyeglasses.xml")
            smile_cascade = cv2.CascadeClassifier(
                "E:\Tool\GitHub\Opencv-master\opencv-master\opencv-master\data\haarcascades\haarcascade_smile.xml")

            # 人脸检测
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 1:  # 如果检测到多个人脸，选择置信度最高的一个
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)  # 按面积排序
                faces = faces[:1]  # 只保留面积最大的一个人脸

            # 绘制人脸矩形
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # 从检测到的人脸区域截取图片进行眼睛和微笑检测
                roi_gray = gray[y:y + h, x:x + w]

                # 眼睛检测，限制为两个
                eyes = eye_cascade.detectMultiScale(roi_gray)
                eyes = eyes[:2]  # 限制结果最多为两个眼睛

                # 绘制眼睛矩形
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(img, (ex + x, ey + y), (ex + x + ew, ey + y + eh), (0, 255, 0), 2)

                # 微笑检测，限制为一个
                smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
                smiles = smiles[:1]  # 限制结果最多为一个微笑

                # 绘制微笑矩形
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(img, (sx + x, sy + y), (sx + x + sw, sy + y + sh), (0, 0, 255), 2)

            # 将带有标记的图片转换为 QImage 对象
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 以符合 QImage 格式
            h, w, ch = img.shape
            qt_image = QImage(img.data, w, h, QImage.Format_RGB888)

            # 调整图片大小以适应标签，同时保持纵横比
            scaled_image = qt_image.scaled(981, 581, Qt.KeepAspectRatio)
            scaled_image = scaled_image.convertToFormat(QImage.Format_ARGB32)  # 确保图像不是灰白的

            # 显示在 label_3 中
            self.label_3.setPixmap(QPixmap.fromImage(scaled_image))

    def facevideo(self):
        # 弹出文件选择对话框，选择视频
        filename, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if filename:
            # 使用 OpenCV 读取视频
            cap = cv2.VideoCapture(filename)
            if not cap.isOpened():
                QMessageBox.warning(self, "警告", "无法打开视频文件，请选择有效的视频文件。")
                return

            # 加载 Haar 特征级联文件
            face_cascade = cv2.CascadeClassifier("E:\Tool\GitHub\Opencv-master\opencv-master\opencv-master\data\haarcascades\haarcascade_frontalface_alt.xml")
            eye_cascade = cv2.CascadeClassifier("E:\Tool\GitHub\Opencv-master\opencv-master\opencv-master\data\haarcascades\haarcascade_eye_tree_eyeglasses.xml")
            smile_cascade = cv2.CascadeClassifier("E:\Tool\GitHub\Opencv-master\opencv-master\opencv-master\data\haarcascades\haarcascade_smile.xml")

            while True:
                # 读取视频的下一帧
                ret, frame = cap.read()
                if not ret:
                    break

                # 将图片转换为灰度图，用于检测
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 人脸检测
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) > 1:
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)  # 按面积排序
                    faces = faces[:1]  # 只保留面积最大的一个人脸

                # 逐个处理检测到的人脸
                for (x, y, w, h) in faces:
                    # 绘制人脸矩形
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # 从检测到的人脸区域截取图片进行眼睛和微笑检测
                    roi_gray = gray[y:y + h, x:x + w]

                    # 眼睛检测，限制为两个
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    eyes = eyes[:2]  # 限制结果最多为两个眼睛

                    # 绘制眼睛矩形
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(frame, (ex + x, ey + y), (ex + x + ew, ey + y + eh), (0, 255, 0), 2)

                    # 微笑检测，限制为一个
                    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
                    smiles = smiles[:1]  # 限制结果最多为一个微笑

                    # 绘制微笑矩形
                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(frame, (sx + x, sy + y), (sx + x + sw, sy + y + sh), (0, 0, 255), 2)

                # 将带有标记的视频帧转换为 QImage 对象
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为 RGB 以符合 QImage 格式
                h, w, ch = frame.shape
                qt_image = QImage(frame.data, w, h, QImage.Format_RGB888)

                # 显示在 label_3 中
                self.label_3.setPixmap(QPixmap.fromImage(qt_image))

                # 按 'q' 退出循环
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 释放视频对象
            cap.release()

    def facecamera(self):
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, "警告", "无法打开摄像头。")
            return

        # 加载 Haar 特征级联文件
        face_cascade = cv2.CascadeClassifier(
            "E:\\Tool\\GitHub\\Opencv-master\\opencv-master\\opencv-master\\data\\haarcascades\\haarcascade_frontalface_alt.xml"
        )
        eye_cascade = cv2.CascadeClassifier(
            "E:\\Tool\\GitHub\\Opencv-master\\opencv-master\\opencv-master\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml"
        )
        smile_cascade = cv2.CascadeClassifier(
            "E:\\Tool\\GitHub\\Opencv-master\\opencv-master\\opencv-master\\data\\haarcascades\\haarcascade_smile.xml"
        )

        try:
            while True:
                # 读取摄像头的下一帧
                ret, frame = cap.read()
                if not ret:
                    break

                # 将图片转换为灰度图，用于检测
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 人脸检测
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                # 限制只处理一个人脸
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # 从检测到的人脸区域截取图片进行眼睛和微笑检测
                    roi_gray = gray[y:y + h, x:x + w]

                    # 眼睛检测
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    if len(eyes) == 2:  # 限制两个眼睛
                        for (ex, ey, ew, eh) in eyes:
                            cv2.rectangle(frame, (ex + x, ey + y), (ex + x + ew, ey + y + eh), (0, 255, 0), 2)

                    # 微笑检测
                    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
                    if len(smiles) > 0:  # 限制一个微笑
                        (sx, sy, sw, sh) = smiles[0]
                        cv2.rectangle(frame, (sx + x, sy + y), (sx + x + sw, sy + y + sh), (0, 0, 255), 2)




                # 显示处理后的帧
                cv2.imshow('Face Camera', frame)

                # 按 'q' 退出循环
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print("Error during face detection from camera:", e)
            QMessageBox.warning(self, "错误", "面部检测出错。")

        finally:
            # 释放摄像头资源
            cap.release()
            # 关闭所有 OpenCV 窗口
            cv2.destroyAllWindows()

    def QRimage(self):
        # 弹出文件选择对话框，选择图片
        filename, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            try:
                # 使用PIL打开图像文件
                pil_image = Image.open(filename)

                # 将PIL图像转换为OpenCV格式
                cv_image = np.array(pil_image)
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)  # PIL图像默认是RGB，OpenCV默认是BGR

                # 使用pyzbar库进行二维码识别
                decoded_objects = pyzbar.decode(cv_image)

                # 初始化解码数据列表
                decoded_data = []

                # 遍历解码对象，提取数据
                for obj in decoded_objects:
                    if obj.type == "QRCODE":
                        # 解码数据并添加到列表中
                        decoded_data.append(obj.data.decode('utf-8'))

                # 准备显示结果的文本
                if decoded_data:
                    result_text = "<html><head/><body><p align=\"center\">识别结果:</p></body></html>" + "".join(
                        f"<p align=\"center\">{data}</p>" for data in decoded_data
                    )
                else:
                    result_text = "<html><head/><body><p align=\"center\">未识别到二维码</p></body></html>"

                # 更新label显示识别结果
                self.label.setText(result_text)

                # 使用cv2.imshow显示图片
                cv2.imshow('QR Code Image', cv_image)
                cv2.waitKey(0)  # 等待用户按键

            except IOError as e:
                # 如果文件打开失败，显示错误消息
                QMessageBox.warning(self, "错误", f"无法打开图片文件: {e}")
            except Exception as e:
                # 显示其他可能的错误
                QMessageBox.warning(self, "错误", f"二维码识别出错: {e}")
            finally:
                # 关闭所有OpenCV窗口
                cv2.destroyAllWindows()

    def QRvideo(self):
        # 弹出文件选择对话框，选择视频
        filename, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if filename:
            # 使用 OpenCV 读取视频
            cap = cv2.VideoCapture(filename)
            if not cap.isOpened():
                QMessageBox.warning(self, "警告", "无法打开视频文件，请选择有效的视频文件。")
                return

            previous_data = None  # 用于存储上一次的识别结果
            decoded_data = []  # 用于存储解码结果

            try:
                # 创建一个窗口来显示视频
                cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

                while True:
                    # 读取视频的下一帧
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 将BGR帧转换为PIL Image
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(image)

                    # 使用pyzbar库进行二维码识别
                    decoded_objects = pyzbar.decode(img)

                    # 检查是否识别到二维码
                    if decoded_objects:
                        for obj in decoded_objects:
                            if obj.type == "QRCODE":
                                # 获取解码的数据
                                data = obj.data.decode('utf-8')
                                # 如果当前识别结果与上一个不同，则更新显示结果
                                if data != previous_data:
                                    previous_data = data
                                    decoded_data.append(data)

                    # 显示当前帧
                    cv2.imshow('Video', frame)

                    # 按 'q' 退出循环
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # 释放视频对象
                cap.release()
                # 关闭所有OpenCV窗口
                cv2.destroyAllWindows()

                # 将识别结果按照换行符输出
                if decoded_data:
                    result_text = "<html><head/><body><p align=\"center\">识别结果:</p></body></html>" + "\n".join(
                        f"<p align=\"center\">{data}</p>" for data in set(decoded_data)  # 使用 set 来去除重复项
                    )
                    self.label.setText(result_text)
                else:
                    self.label.setText("<html><head/><body><p align=\"center\">视频中未识别到二维码</p></body></html>")

            except Exception as e:
                print("Error during QR code decoding from video:", e)
                self.label.setText("<html><head/><body><p align=\"center\">二维码识别出错</p></body></html>")



    def QRcamera(self):
        # 打开摄像头 按下q可以退出摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, "警告", "无法打开摄像头。")
            return

        try:
            while True:
                # 读取摄像头的下一帧
                ret, frame = cap.read()
                if not ret:
                    break

                # 将 BGR 帧转换为 PIL Image
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(image)

                # 使用pyzbar库进行二维码识别
                decoded_objects = pyzbar.decode(img)

                # 检查是否识别到二维码
                if decoded_objects:
                    decoded_data = [obj.data.decode('utf-8') for obj in decoded_objects if obj.type == 'QRCODE']
                    if decoded_data:
                        # 如果识别到二维码，将结果按照换行符输出
                        result_text = "<html><head/><body><p align=\"center\">识别结果:</p></body></html>" + \
                                      "\n".join(f"<p align=\"center\">{data}</p>" for data in decoded_data)
                        self.label.setText(result_text)
                    else:
                        self.label.setText("<html><head/><body><p align=\"center\">未识别到二维码</p></body></html>")
                else:
                    self.label.setText("<html><head/><body><p align=\"center\">未识别到二维码</p></body></html>")

                # 显示处理后的帧
                cv2.imshow('QR Camera', frame)

                # 按 'q' 退出循环
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print("Error during QR code decoding from camera:", e)
            self.label.setText("<html><head/><body><p align=\"center\">二维码识别出错</p></body></html>")

        finally:
            # 释放摄像头资源
            cap.release()
            # 关闭所有 OpenCV 窗口
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())