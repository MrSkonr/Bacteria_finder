# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\Bacteria_finder_main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap

import sys
sys.path.append('../Bacteria_finder')

from Annotation_functions import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 710)
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        MainWindow.setFont(font)
        MainWindow.setWindowOpacity(3.0)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.LoadImageButton = QtWidgets.QPushButton(self.centralwidget)
        self.LoadImageButton.setGeometry(QtCore.QRect(850, 640, 131, 41))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.LoadImageButton.setFont(font)
        self.LoadImageButton.setObjectName("LoadImageButton")
        self.SaveImageButton = QtWidgets.QPushButton(self.centralwidget)
        self.SaveImageButton.setEnabled(False)
        self.SaveImageButton.setGeometry(QtCore.QRect(1100, 640, 131, 41))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.SaveImageButton.setFont(font)
        self.SaveImageButton.setObjectName("SaveImageButton")
        self.ColorSpaceGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.ColorSpaceGroupBox.setEnabled(False)
        self.ColorSpaceGroupBox.setGeometry(QtCore.QRect(810, 10, 461, 91))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.ColorSpaceGroupBox.setFont(font)
        self.ColorSpaceGroupBox.setObjectName("ColorSpaceGroupBox")
        self.ChannelsComboBox = QtWidgets.QComboBox(self.ColorSpaceGroupBox)
        self.ChannelsComboBox.setGeometry(QtCore.QRect(30, 30, 181, 41))
        self.ChannelsComboBox.setObjectName("ChannelsComboBox")
        self.ChannelsComboBox.addItem("")
        self.ChannelsComboBox.addItem("")
        self.ChannelsComboBox.addItem("")
        self.ChannelsComboBox.addItem("")
        self.ChannelsComboBox.addItem("")
        self.ChannelsComboBox.addItem("")
        self.ChannelsComboBox.addItem("")
        self.ChannelsComboBox.addItem("")
        self.SetColorSpaceButton = QtWidgets.QPushButton(self.ColorSpaceGroupBox)
        self.SetColorSpaceButton.setGeometry(QtCore.QRect(260, 30, 171, 41))
        self.SetColorSpaceButton.setObjectName("SetColorSpaceButton")
        self.ImageGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.ImageGroupBox.setGeometry(QtCore.QRect(9, 0, 791, 690))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.ImageGroupBox.setFont(font)
        self.ImageGroupBox.setObjectName("ImageGroupBox")
        self.ImageLabel = QtWidgets.QLabel(self.ImageGroupBox)
        self.ImageLabel.setGeometry(QtCore.QRect(10, 25, 771, 655))
        self.ImageLabel.setText("")
        self.ImageLabel.setPixmap(QtGui.QPixmap(".\\blank.png"))
        self.ImageLabel.setScaledContents(True)
        self.ImageLabel.setObjectName("ImageLabel")
        self.ThresholdingGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.ThresholdingGroupBox.setEnabled(False)
        self.ThresholdingGroupBox.setGeometry(QtCore.QRect(810, 110, 461, 111))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.ThresholdingGroupBox.setFont(font)
        self.ThresholdingGroupBox.setObjectName("ThresholdingGroupBox")
        self.SetThresholdButton = QtWidgets.QPushButton(self.ThresholdingGroupBox)
        self.SetThresholdButton.setGeometry(QtCore.QRect(310, 70, 141, 31))
        self.SetThresholdButton.setObjectName("SetThresholdButton")
        self.ThresholdSlider = QtWidgets.QSlider(self.ThresholdingGroupBox)
        self.ThresholdSlider.setGeometry(QtCore.QRect(10, 30, 441, 21))
        self.ThresholdSlider.setMaximum(255)
        self.ThresholdSlider.setProperty("value", 127)
        self.ThresholdSlider.setOrientation(QtCore.Qt.Horizontal)
        self.ThresholdSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.ThresholdSlider.setTickInterval(5)
        self.ThresholdSlider.setObjectName("ThresholdSlider")
        self.ThresholdSpinBox = QtWidgets.QSpinBox(self.ThresholdingGroupBox)
        self.ThresholdSpinBox.setGeometry(QtCore.QRect(230, 70, 71, 31))
        self.ThresholdSpinBox.setMaximum(255)
        self.ThresholdSpinBox.setProperty("value", 127)
        self.ThresholdSpinBox.setObjectName("ThresholdSpinBox")
        self.BoundingBoxesGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.BoundingBoxesGroupBox.setEnabled(False)
        self.BoundingBoxesGroupBox.setGeometry(QtCore.QRect(1040, 570, 231, 61))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.BoundingBoxesGroupBox.setFont(font)
        self.BoundingBoxesGroupBox.setObjectName("BoundingBoxesGroupBox")
        self.DrawBoundingBoxesCheckBox = QtWidgets.QCheckBox(self.BoundingBoxesGroupBox)
        self.DrawBoundingBoxesCheckBox.setGeometry(QtCore.QRect(10, 30, 211, 21))
        self.DrawBoundingBoxesCheckBox.setObjectName("DrawBoundingBoxesCheckBox")
        self.ReverseToOriginalButton = QtWidgets.QPushButton(self.centralwidget)
        self.ReverseToOriginalButton.setEnabled(False)
        self.ReverseToOriginalButton.setGeometry(QtCore.QRect(1090, 230, 171, 41))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.ReverseToOriginalButton.setFont(font)
        self.ReverseToOriginalButton.setObjectName("ReverseToOriginalButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # All the connections are below
        self.LoadImageButton.clicked.connect(self.LoadImageButtonPushed)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Bacteria finder"))
        self.LoadImageButton.setText(_translate("MainWindow", "Load image"))
        self.SaveImageButton.setText(_translate("MainWindow", "Save image"))
        self.ColorSpaceGroupBox.setTitle(_translate("MainWindow", "Color space"))
        self.ChannelsComboBox.setItemText(0, _translate("MainWindow", "Original image"))
        self.ChannelsComboBox.setItemText(1, _translate("MainWindow", "Gray-scale image"))
        self.ChannelsComboBox.setItemText(2, _translate("MainWindow", "Red channel"))
        self.ChannelsComboBox.setItemText(3, _translate("MainWindow", "Green channel"))
        self.ChannelsComboBox.setItemText(4, _translate("MainWindow", "Blue channel"))
        self.ChannelsComboBox.setItemText(5, _translate("MainWindow", "Hue channel"))
        self.ChannelsComboBox.setItemText(6, _translate("MainWindow", "Sat channel"))
        self.ChannelsComboBox.setItemText(7, _translate("MainWindow", "Val channel"))
        self.SetColorSpaceButton.setText(_translate("MainWindow", "Set color space"))
        self.ImageGroupBox.setTitle(_translate("MainWindow", "Image"))
        self.ThresholdingGroupBox.setTitle(_translate("MainWindow", "Thresholding"))
        self.SetThresholdButton.setText(_translate("MainWindow", "Set threshold"))
        self.BoundingBoxesGroupBox.setTitle(_translate("MainWindow", "Bounding boxes"))
        self.DrawBoundingBoxesCheckBox.setText(_translate("MainWindow", "Draw bounding boxes"))
        self.ReverseToOriginalButton.setText(_translate("MainWindow", "Reverse to original"))

    def LoadImageButtonPushed(self):
        '''
        Function, that: 
        1) loads the image
        2) saves the image in cv2
        3) displays it in application
        '''
        # Loading the image
        self.File_name = QtWidgets.QFileDialog.getOpenFileName(self.ImageLabel, 'Open file', '', "Image files (*.png *.jpg *.bmp *.gif)")
        self.File_path = self.File_name[0]

        # Check if the user didn't choose the file
        if self.File_path == '':
            return

        # Saving the image for future use in cv2
        self.original_bacteria_image = cv2.imread(self.File_path)
        self.changed_bacteria_image = select_colorsp(self.original_bacteria_image, 'gray')
        self.changed_bounded_bacteria_image = select_colorsp(self.original_bacteria_image, 'gray')

        # Displaying the image
        pixmap = QPixmap(self.File_path)
        self.ImageLabel.setPixmap(QPixmap(pixmap))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
