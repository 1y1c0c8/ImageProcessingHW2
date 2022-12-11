from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog

import os
import cv2
import numpy as np
import glob

from UI import Ui_MainWindow

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.path1 = ""
        self.path2 = ""
        self.dirPath = ""
        self.char_in_board = [  # coordinate for 6 charter in board (x, y) ==> (w, h)
            [7, 5, 0],  # slot 1
            [4, 5, 0],  # slot 2
            [1, 5, 0],  # slot 3
            [7, 2, 0],  # slot 4
            [4, 2, 0],  # slot 5
            [1, 2, 0]  # slot 6
        ]
        self.base = [
            58, 34, 10, 61, 37, 13
        ]

        # self.mtx = np.array([[2.22484480e+03, 0.00000000e+00, 1.03009432e+03],
        #                 [0.00000000e+00, 2.22404212e+03, 1.03961767e+03],
        #                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        #
        # self.dist = np.array([[-0.12541579, 0.07937593, -0.00072728, 0.0005207, 0.01956878]])


    # Connection
    def setup_control(self):
        self.ui.LoadFolderButton.clicked.connect(self.loadFolder)
        self.ui.LoadImageLButton.clicked.connect(self.loadImgL)
        self.ui.LoadImageRButton.clicked.connect(self.loadImgR)

        self.ui.DrawContourButton.clicked.connect(self.drawContour)
        self.ui.CountRingButton.clicked.connect(self.countRing)

        self.ui.findCornerButton.clicked.connect(self.findCorner)
        self.ui.findIntrinsicButton.clicked.connect(self.findParamAndPrintIntrinsic)
        self.ui.findExtrinsicButton.clicked.connect(self.printExtrinsic)
        self.ui.findDistortionButton.clicked.connect(self.printDistortion)
        self.ui.showResultButton.clicked.connect(self.showResult)
        self.ui.showWordsOnBoardButton.clicked.connect(self.showWordsOnBoard)
        self.ui.shoWordsVerticallyButton.clicked.connect(self.Show_Words_Vertically)
        self.ui.showDisparityMapButton.clicked.connect(self.stereoDisparityMap)
    #Functions
    def loadFolder(self):
        self.dirPath = QFileDialog.getExistingDirectory()
        self.ui.loadFolderLabel.setText(self.dirPath)
        self.dirPath = os.path.join(self.dirPath, "*.bmp")

    def loadImgL(self):
        self.path1, filetype = QFileDialog.getOpenFileName(self,
                                                          "Open file",
                                                          self.dirPath)
        filename = os.path.basename(self.path1)
        self.ui.loadImageLLabel.setText(filename)

    def loadImgR(self):
        self.path2, filetype = QFileDialog.getOpenFileName(self,
                                                          "Open file",
                                                          self.dirPath)
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

        self.canny1 = cv2.resize(self.canny1, (width1, height1))
        self.canny2 = cv2.resize(self.canny2, (width2, height2))

        cv2.imshow('Result1', self.canny1)
        cv2.imshow('Result2', self.canny2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def countRing(self):
        img1 = cv2.imread(self.path1)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
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
        blur2 = cv2.medianBlur(gray2, 5)

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
        chess_images = glob.glob(self.dirPath)
        # print(len(chess_images))

        for i in range(len(chess_images)):
            # time.sleep(0.5)
            chess_board_image = cv2.imread(chess_images[i])
            gray = cv2.cvtColor(chess_board_image, cv2.COLOR_BGR2GRAY)

            ny = 8
            nx = 11
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

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

    def findParam(self):
        chess_images = glob.glob(self.dirPath)
        # define criteria = (type, max_iter, epsilon)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane

        for i in range(len(chess_images)):
            # Read in the image
            image = cv2.imread(chess_images[i])
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)

            if ret == True:
                self.objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
                self.imgpoints.append(corners2)

        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)


        str_inputNum = self.ui.comboBox.currentText()
        int_inputNum = int(str_inputNum)
        R = cv2.Rodrigues(self.rvecs[int_inputNum - 1])
        self.ext = np.hstack((R[0], self.tvecs[int_inputNum - 1]))


        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def findParamAndPrintIntrinsic(self):
        self.findParam()
        print(f'Intrinsic matrix:\n{self.mtx}')

    def printExtrinsic(self):

        print(f'Extrinsic matrix:\n{self.ext}')

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def printDistortion(self):
        print(f'Distortion:\n{self.dist}')

        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def showResult(self):

        chess_images = glob.glob(self.dirPath)

        for i in range(len(chess_images)):
            img = cv2.imread(chess_images[i])
            h, w = img.shape[:2]
            newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
            dst = cv2.undistort(img, self.mtx, self.dist, None, newCameraMatrix)

            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]

            img = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2))
            dst = cv2.resize(img, (img.shape[0], img.shape[1]))

            result = np.hstack((img, dst))
            cv2.imshow("Result", result)
            cv2.waitKey(500)
            cv2.destroyWindow('Result')


        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def getEndpointsCoordinate(self, line):
        # return coordinates of a line's 2 endpoints
        co = (
            (int(line[0][0]), int(line[0][1])),
            (int(line[1][0]), int(line[1][1]))
        )

        return co

    def getLineList(self, alphabet):
        # 他媽的記得把fs改成self.
        ch = self.fs.getNode(alphabet).mat()

        lineList = []

        for line in range(len(ch)):
            lineList.append(ch[line])

        return lineList

    def baseToIndex(self, co, base):
        # index = 8*co[0] - col[1]

        index = 0
        if co == (0,2):
            index = base-2
        elif co == (0,1):
            index = base-1
        elif co == (0,0):
            index = base+0
        elif co == (1,2):
            index = base+6
        elif co == (1,1):
            index = base+7
        elif co == (1,0):
            index = base+8
        elif co == (2,2):
            index = base+14
        elif co == (2,1):
            index = base+15
        elif co == (2,0):
            index = base+16

        return index
    def drawLine(self, img, imgIndex, indexStart, indexEnd):
        x1 = int(tuple(self.imgpoints[imgIndex][indexStart][0])[0] // 4)
        y1 = int(tuple(self.imgpoints[imgIndex][indexStart][0])[1] // 4)
        # print(indexStart)
        # print((x1, y1))

        x2 = int(tuple(self.imgpoints[imgIndex][indexEnd][0])[0] // 4)
        y2 = int(tuple(self.imgpoints[imgIndex][indexEnd][0])[1] // 4)
        # print(indexEnd)
        # print((x2, y2))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)

    def drawAnAlphabet(self, img, imgIndex, alphabet, alphabetIndex):
        lineList = self.getLineList(alphabet)

        for line in lineList:
            # print('before getEndpointsCoordinate')
            COs = self.getEndpointsCoordinate(line)
            # print('after getEndpointsCoordinate')
            base = self.base[alphabetIndex]
            print(COs[0])
            print(COs[1])
            index0 = self.baseToIndex(COs[0], base)
            index1 = self.baseToIndex(COs[1], base)


            self.drawLine(img, imgIndex, index0, index1)


    def showWordsOnBoard(self):

        # 1.Calibrate: get Intrinsic(self.mtx), distortion(self.dist), extrinsic(self.ext)
        self.findParam()

        # 2.Input a “Word” less than 6 char in English in the textEdit box
        self.string = self.ui.lineEdit.text()[:6].upper()

        # 3.Derive the shape of the “Word” by using the provided library
        self.fs = cv2.FileStorage('Q3_Image/Q2_lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)

        chess_images = glob.glob(self.dirPath)

        for imgIndex in range(len(chess_images)):

            alphabetIndex = 0

            img = cv2.imread(chess_images[imgIndex])
            img = cv2.resize(img, (img.shape[0] // 4, img.shape[1] // 4))

            for alphabet in self.string:
                self.drawAnAlphabet(img, imgIndex, alphabet, alphabetIndex)
                alphabetIndex = alphabetIndex+1

            cv2.imshow('result', img)
            cv2.waitKey(1000)
            cv2.destroyWindow('result')
            print('==========================')

        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def showWordsVertically(self):
        # # termination criteria
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        #
        # # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # objp = np.zeros((11 * 8, 3), np.float32)
        # objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)
        #
        # # axis = np.float32([[3, 3, -3], [1, 1, 0], [3, 5, 0], [5, 1, 0]]).reshape(-1, 3)
        # axis = np.float32([[5, 3, -3], [7, 1, 0], [3, 3, 0], [7, 5, 0]]).reshape(-1, 3)
        #
        # # Arrays to store object points and image points from all the images.
        # objpoints = []  # 3d point in real world space
        # imgpoints = []  # 2d points in image plane
        #
        # print(self.dirPath)
        # chess_images = glob.glob(self.dirPath)
        # # Select any index to grab an image from the list
        # for i in range(len(chess_images)):
        #     # Read in the image
        #     image = cv2.imread(chess_images[i])
        #     # Convert to grayscale
        #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #     # Find the chessboard corners
        #     ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)
        #
        #     if ret == True:
        #         objpoints.append(objp)
        #         # objp = 8 * 11 objpoints (x, y, z)
        #         corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
        #         imgpoints.append(corners2)
        #         # corner2 = each object point on 2D image (x, y)
        #
        #         # gray.shape[::-1] = (2048, 2048)
        #         ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)
        #
        #         # project 3D points to image plane
        #         imgpts, jac = cv2.projectPoints(axis, rvecs[i], tvecs[i], mtx, dist)
        #
        #         def draw(image, imgpts):
        #             image = cv2.line(image, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), 5)
        #             image = cv2.line(image, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        #             image = cv2.line(image, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (0, 0, 255), 5)
        #             image = cv2.line(image, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        #             image = cv2.line(image, tuple(imgpts[1].ravel()), tuple(imgpts[3].ravel()), (0, 0, 255), 5)
        #             image = cv2.line(image, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0, 0, 255), 5)
        #             return image
        #
        #         img = draw(image, imgpts)
        #
        #         cv2.imwrite('%s_v.jpg' % i, img)
        #         img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
        #         cv2.namedWindow('img')
        #         cv2.imshow('img', img)
        #         cv2.waitKey(1000)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Show_Words_Vertically(self):  # input for the text
        # # 1.Calibrate: get Intrinsic(self.mtx), distortion(self.dist), extrinsic(self.ext)
        # self.findParam()
        #
        # self.string = self.ui.lineEdit.text()[:6].upper()
        #
        # self.fs = cv2.FileStorage('Q3_Image/Q2_lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
        #
        # chess_images = glob.glob(self.dirPath)
        #
        # for imgIndex in range(len(chess_images)):
        #
        #     alphabetIndex = 0
        #
        #     img = cv2.imread(chess_images[imgIndex])
        #
        #     # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, (2048, 2048), None, None)
        #
        #     for imgIndex in range(len(self.string)):
        #         file = cv2.FileStorage('Q3_Image/Q2_lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
        #
        #         ch = np.float32(file.getNode(input[imgIndex].upper()).mat()).reshape(-1, 3)
        #         for k in range(len(ch)):  # Change x and y
        #             temp = ch[k][0]
        #             ch[k][0] = ch[k][1]
        #             ch[k][1] = temp
        #         for k in range(len(ch)):  # Upside down
        #             if ch[k][0] == 2:
        #                 ch[k][0] = 0
        #             elif ch[k][0] == 0:
        #                 ch[k][0] = 2
        #         if imgIndex == 0:
        #             for k in range(len(ch)):
        #                 ch[k][1] += 8
        #         elif imgIndex == 1:
        #             for k in range(len(ch)):
        #                 ch[k][1] += 4
        #         elif imgIndex == 2:
        #             pass
        #         elif imgIndex == 3:
        #             for k in range(len(ch)):
        #                 ch[k][1] += 8
        #                 ch[k][0] += 4
        #         elif imgIndex == 4:
        #             for k in range(len(ch)):
        #                 ch[k][1] += 4
        #                 ch[k][0] += 4
        #         elif imgIndex == 5:
        #             for k in range(len(ch)):
        #                 ch[k][0] += 4
        #
        #         print(ch)
        #         imgpts, jac = cv2.projectPoints(ch, self.rvecs[imgIndex], self.tvecs[imgIndex], self.mtx, self.dist)
        #         for k in range(0, len(ch), 2):
        #             image = cv2.line(image, tuple(imgpts[k].ravel()), tuple(imgpts[k + 1].ravel()), (0, 0, 255), 5)
        #
        #     cv2.namedWindow("Augmented Reality", cv2.WINDOW_GUI_EXPANDED)
        #     # image = cv2.resize(image, (image.shape[0]//4, image.shape[1]//4))
        #     cv2.imshow("Augmented Reality", image)
        #     cv2.waitKey(1000)
        #
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def stereoDisparityMap(self):
        img1 = cv2.imread(self.path1)
        img2 = cv2.imread(self.path2)

        stereo = cv2.StereoBM_create(numDisparities = 256, blockSize=25)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(img1_gray, img2_gray).astype(np.float32) /16.0
        disparity_show = cv2.normalize(disparity, None, 0, 1, cv2.NORM_MINMAX)
        cv2.imshow('Result', disparity_show)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
