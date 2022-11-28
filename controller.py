from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog

import os
from PIL import Image
import cv2
import numpy as np
from tkinter import filedialog
import glob
import time

from UI import Ui_MainWindow

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.path1 = ""
        self.path2 = ""
        self.folderPath = ""


    # Connection
    def setup_control(self):
        self.ui.LoadFolderButton.clicked.connect(self.loadFolder)
        self.ui.LoadImageLButton.clicked.connect(self.loadImgL)
        self.ui.LoadImageRButton.clicked.connect(self.loadImgR)

        self.ui.DrawContourButton.clicked.connect(self.drawContour)
        self.ui.CountRingButton.clicked.connect(self.countRing)

        self.ui.findCornerButton.clicked.connect(self.findCorner)


    #Functions
    def loadFolder(self):

        # self.dirPath = filedialog.askdirectory(initialdir="C:\\Users\kenny\PycharmProjects\ImageProcessingHw2", title="Select directory")
        # self.ui.loadFolderLabel.setText(self.dirPath)
        # self.dirPath += '/*.bmp'
    #     ============================
        self.dirPath = QFileDialog.getExistingDirectory()
        self.ui.loadFolderLabel.setText(self.dirPath)
        print(self.dirPath)


    def loadImgL(self):
        self.path1, filetype = QFileDialog.getOpenFileName(self,
                                                          "Open file",
                                                          ".")
        filename = os.path.basename(self.path1)
        self.ui.loadImageLLabel.setText(filename)

    def loadImgR(self):
        self.path2, filetype = QFileDialog.getOpenFileName(self,
                                                          "Open file",
                                                          ".")
        filename = os.path.basename(self.path2)
        self.ui.loadImageRLabel.setText(filename)

    def drawContour(self):
        img1 = cv2.imread(self.path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(self.path2, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('Input1', img1)
        cv2.imshow('Input2', img2)

        height1 = img1.shape[0]//2
        width1 = img1.shape[1]//2
        height2 = img2.shape[0]//2
        width2 = img2.shape[1]//2

        img1 = cv2.resize(img1, (width1, height1))
        img2 = cv2.resize(img2, (width2, height2))

        blurred1 = cv2.GaussianBlur(img1, (5, 5), 0)
        self.canny1 = cv2.Canny(blurred1, 30, 150)
        blurred2 = cv2.GaussianBlur(img2, (5, 5), 0)
        self.canny2 = cv2.Canny(blurred2, 30, 150)

        # height1 = (self.canny1.shape[0])*2
        # width1 = (self.canny1.shape[1])*2
        # height2 = (self.canny2.shape[0])*2
        # width2 = (self.canny2.shape[1])*2

        self.canny1 = cv2.resize(self.canny1, (width1, height1))
        self.canny2 = cv2.resize(self.canny2, (width2, height2))

        cv2.imshow('Result1', self.canny1)
        cv2.imshow('Result2', self.canny2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def countRing(self):
        img1 = cv2.imread(self.path1)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray, (5, 5), 0)
        blur1 = cv2.medianBlur(gray1, 5)
        cimg1 = cv2.cvtColor(blur1, cv2.COLOR_GRAY2BGR)

        circles1 = cv2.HoughCircles(blur1, cv2.HOUGH_GRADIENT, 1, 120, param1=100, param2=30, minRadius=0, maxRadius=0)
        circles1 = np.uint16(np.around(circles1))
        for i in circles1[0, :]:

            #Outer circle
            cv2.circle(img1, (i[0], i[1]), i[2], (0, 255,0), 2)

            #Center of the circle
            cv2.circle(img1, (i[0], i[1]), 2, (0, 255, 0), 3)

        cv2.imshow("circle detection", img1)

        img2 = cv2.imread(self.path2)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray, (5, 5), 0)
        blur2 = cv2.medianBlur(gray2, 5)
        cimg2 = cv2.cvtColor(blur2, cv2.COLOR_GRAY2BGR)

        circles2 = cv2.HoughCircles(blur2, cv2.HOUGH_GRADIENT, 1, 120, param1=100, param2=30, minRadius=0, maxRadius=0)
        circles2 = np.uint16(np.around(circles2))
        for i in circles2[0, :]:
            # Outer circle
            cv2.circle(img2, (i[0], i[1]), i[2], (0, 255, 0), 2)

            # Center of the circle
            cv2.circle(img2, (i[0], i[1]), 2, (0, 255, 0), 3)

        cv2.imshow("circle detection2", img2)

        self.ui.countResultLabel.setText("There are " + str(len(circles1[0])) + " rings in img1.jpg.\n" + "There are " + str(len(circles2[0])) + " rings in img2.jpg")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def findCorner(self):
        # print(self.dirPath)
        chess_images = glob.glob('./Q2_Image/*.bmp')
        print(len(chess_images))

        for i in range(len(chess_images)):
            # time.sleep(0.5)
            chess_board_image = cv2.imread(chess_images[i])
            gray = cv2.cvtColor(chess_board_image, cv2.COLOR_BGR2GRAY)

            ny = 8
            nx = 11
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            # print('ret: ', ret)

            if ret == True:
                cv2.drawChessboardCorners(chess_board_image, (nx, ny), corners, ret)
                # result_name = 'board' + str(i + 1) + '.bmp'
                # cv2.imwrite(result_name, chess_board_image)
                # cv2.namedWindow("%s" % (i + 1))
                # cv2.imshow("%s" %/ (i + 1), chess_board_image)
                chess_board_image = cv2.resize(chess_board_image, (chess_board_image.shape[0]//4, chess_board_image.shape[1]//4))
                # cv2.imshow("Display", chess_board_image)
                # print('sleep')
                chess_images[i] = chess_board_image

        for i in range(len(chess_images)):
            cv2.imshow("Display", chess_images[i])
            cv2.waitKey(500)
            cv2.destroyWindow('Display')

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # def findIntrinsic():

    # def findExtrinsic():

    # def findDistortion():

    # def showResult():

    # def showWordsOnBoard():

    # def showWordsVertically():

    # def stereoDisparityMap():

