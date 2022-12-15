# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\Bacteria_finder_main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage

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
        self.InverseBinaryCheckBox = QtWidgets.QCheckBox(self.ThresholdingGroupBox)
        self.InverseBinaryCheckBox.setGeometry(QtCore.QRect(10, 80, 151, 21))
        self.InverseBinaryCheckBox.setObjectName("InverseBinaryCheckBox")
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
        self.DrawBoundingBoxesCheckBox.setTristate(False)
        self.DrawBoundingBoxesCheckBox.setObjectName("DrawBoundingBoxesCheckBox")
        self.UtilitiesGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.UtilitiesGroupBox.setEnabled(False)
        self.UtilitiesGroupBox.setGeometry(QtCore.QRect(1020, 230, 251, 121))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.UtilitiesGroupBox.setFont(font)
        self.UtilitiesGroupBox.setObjectName("UtilitiesGroupBox")
        self.ReverseToOriginalButton = QtWidgets.QPushButton(self.UtilitiesGroupBox)
        self.ReverseToOriginalButton.setGeometry(QtCore.QRect(70, 70, 171, 41))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.ReverseToOriginalButton.setFont(font)
        self.ReverseToOriginalButton.setObjectName("ReverseToOriginalButton")
        self.ShowOriginalImageCheckBox = QtWidgets.QCheckBox(self.UtilitiesGroupBox)
        self.ShowOriginalImageCheckBox.setGeometry(QtCore.QRect(40, 30, 201, 31))
        self.ShowOriginalImageCheckBox.setObjectName("ShowOriginalImageCheckBox")
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
        self.ChannelsComboBox.activated.connect(self.ChannelsComboBoxChanged)
        self.SetColorSpaceButton.clicked.connect(self.SetColorSpaceButtonPushed)
        self.ThresholdSlider.valueChanged.connect(self.ThresholdSliderChanged)
        self.ThresholdSpinBox.valueChanged.connect(self.ThresholdSpinBoxChanged)
        self.DrawBoundingBoxesCheckBox.stateChanged.connect(self.DrawBoundingBoxesCheckBoxChanged)
        self.SaveImageButton.clicked.connect(self.SaveImageButtonPushed)
        self.ReverseToOriginalButton.clicked.connect(self.ReverseToOriginalImage)
        self.InverseBinaryCheckBox.stateChanged.connect(self.InverseBinaryCheckBoxChanged)
        self.ShowOriginalImageCheckBox.stateChanged.connect(self.ShowOriginalImageCheckBoxChanged)

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
        self.InverseBinaryCheckBox.setText(_translate("MainWindow", "Inverse binary"))
        self.BoundingBoxesGroupBox.setTitle(_translate("MainWindow", "Bounding boxes"))
        self.DrawBoundingBoxesCheckBox.setText(_translate("MainWindow", "Draw bounding boxes"))
        self.UtilitiesGroupBox.setTitle(_translate("MainWindow", "Utilities"))
        self.ReverseToOriginalButton.setText(_translate("MainWindow", "Reverse to original"))
        self.ShowOriginalImageCheckBox.setText(_translate("MainWindow", "Show original image"))

    
    def ShowOriginalImageCheckBoxChanged(self):
        '''
        Calls UpdateImage
        '''
        self.UpdateImage()
    
    def InverseBinaryCheckBoxChanged(self):
        '''
        Calls ThresholdSliderChanged
        '''
        self.ThresholdSliderChanged(self.ThresholdSpinBox.value())
    
    def SaveImageButtonPushed(self):
        '''
        Function, opens save dialog window and saves the shown image
        '''
        # Checking if "Original image" is still chosen
        if self.ChannelsComboBox.currentText() == "Original image":
            warn_msg = QtWidgets.QMessageBox()
            warn_msg.setWindowTitle("Bacteria finder")
            warn_msg.setText("Please choose one of the color channels")
            warn_msg.setIcon(QtWidgets.QMessageBox.Warning)
            warn_msg.exec_()
            return
        
        # Getting the path for the image
        self.File_save_name = QtWidgets.QFileDialog.getSaveFileName(self.ImageLabel, 'Save file', '', "Image files (*.png *.jpg *.bmp *.gif)")
        self.File_save_path = self.File_save_name[0]

        # Check if the user didn't choose the path
        if self.File_save_path == '':
            return

        # Saving in choisen path
        cv2.imwrite(self.File_save_path, self.displayed_image.copy())
    
    def DrawBoundingBoxesCheckBoxChanged(self, value):
        '''
        Function, that switches between drawing bounding boxes or not
        '''
        # Check the value of CheckBox
        if value:
            self.changed_bounded_bacteria_image = draw_annotations(self.changed_bacteria_image.copy(), get_boxes(self.changed_bacteria_image.copy()), thickness= 1)
        else:
            self.changed_bounded_bacteria_image = self.changed_bacteria_image.copy()

        # Updating the shown image
        self.UpdateImage()
    
    def ThresholdSliderChanged(self, value):
        '''
        Function, that performes threshold upon changed value
        '''
        # Dictionary for InverseBinary
        inverse_binary_dict = {True : 'inverse', False:  'direct'}

        # Synchronizing with spin box
        self.ThresholdSpinBox.setValue(value)

        # Thresholding
        if self.DrawBoundingBoxesCheckBox.isChecked():
            self.changed_bacteria_image = threshold(self.channeled_bacteria_image, 
                                                    thresh=value, mode=inverse_binary_dict[self.InverseBinaryCheckBox.isChecked()])
            self.changed_bounded_bacteria_image = draw_annotations(self.changed_bacteria_image.copy(), get_boxes(self.changed_bacteria_image), thickness= 1)
        else:
            self.changed_bacteria_image = threshold(self.channeled_bacteria_image, 
                                                    thresh=value, mode=inverse_binary_dict[self.InverseBinaryCheckBox.isChecked()])
        
        # Updating the shown image
        self.UpdateImage()

    def ThresholdSpinBoxChanged(self, value):
        '''
        Function, that performes threshold upon changed value
        '''
        # Synchronizing with slider
        self.ThresholdSlider.setValue(value)

        # Thresholding
        if self.DrawBoundingBoxesCheckBox.isChecked():
            self.changed_bacteria_image = threshold(self.channeled_bacteria_image, 
                                                    thresh=value, mode='direct')
            self.changed_bounded_bacteria_image = draw_annotations(self.changed_bacteria_image.copy(), get_boxes(self.changed_bacteria_image), thickness= 1)
        else:
            self.changed_bacteria_image = threshold(self.channeled_bacteria_image, 
                                                    thresh=value, mode='direct')
        
        # Updating the shown image
        self.UpdateImage()

    
    def SetColorSpaceButtonPushed(self):
        '''
        Function, that fixates the chosen color channel and activates thresholding and bounding boxing
        Also, the initial thresholding is performed
        '''
        # Checking if "Original image" is still chosen
        if self.ChannelsComboBox.currentText() == "Original image":
            warn_msg = QtWidgets.QMessageBox()
            warn_msg.setWindowTitle("Bacteria finder")
            warn_msg.setText("Please choose one of the color channels")
            warn_msg.setIcon(QtWidgets.QMessageBox.Warning)
            warn_msg.exec_()
            return

        
        # Activating and deactivating
        self.ThresholdingGroupBox.setEnabled(True)
        self.BoundingBoxesGroupBox.setEnabled(True)
        self.ColorSpaceGroupBox.setEnabled(False)
        self.UtilitiesGroupBox.setEnabled(True)

        # Initial Thresholding
        self.channeled_bacteria_image = self.changed_bacteria_image.copy()
        self.changed_bacteria_image = threshold(self.channeled_bacteria_image, 
                                                thresh=self.ThresholdSpinBox.value(), mode='direct')
        self.changed_bounded_bacteria_image = threshold(self.channeled_bacteria_image, 
                                                        thresh=self.ThresholdSpinBox.value(), mode='direct')
        
        # Updating the shown image
        self.UpdateImage()

    
    def ChannelsComboBoxChanged(self, index):
        '''
        Function, that triggeres every time user changes the channel in Group Box
        '''
        # Dictionary for itemText
        itemText_to_key = {"Gray-scale image" : 'gray', "Red channel" : 'red', "Green channel" : 'green', "Blue channel" : 'blue',
                            "Hue channel" : 'hue', "Sat channel" : 'sat', "Val channel" : 'val'}
        
        # Changing stored images to chosen channel
        if self.ChannelsComboBox.itemText(index) == "Original image":
            # Updating the shown image
            self.UpdateImage(True)
        else:
            self.changed_bacteria_image = select_colorsp(self.original_bacteria_image, itemText_to_key[self.ChannelsComboBox.itemText(index)])
            self.changed_bounded_bacteria_image = select_colorsp(self.original_bacteria_image, itemText_to_key[self.ChannelsComboBox.itemText(index)])
            # Updating the shown image
            self.UpdateImage()
    
    def LoadImageButtonPushed(self):
        '''
        Function, that loads the path to image
        '''
        # Loading the image
        self.File_load_name = QtWidgets.QFileDialog.getOpenFileName(self.ImageLabel, 'Open file', '', "Image files (*.png *.jpg *.bmp *.gif)")
        self.File_load_path = self.File_load_name[0]

        # Check if the user didn't choose the file
        if self.File_load_path == '':
            return

        # Calls the reverse function
        self.ReverseToOriginalImage()

    def ReverseToOriginalImage(self):
        '''
        Function, that reverses all changes made by user and reloads the image
        from previously chosen folder
        '''
        # Saving the image for future use in cv2
        self.original_bacteria_image = cv2.imread(self.File_load_path)
        self.changed_bacteria_image = select_colorsp(self.original_bacteria_image, 'gray')
        self.changed_bounded_bacteria_image = select_colorsp(self.original_bacteria_image, 'gray')

        # Enabling and disabling widgets
        self.ColorSpaceGroupBox.setEnabled(True)
        self.ThresholdingGroupBox.setEnabled(False)
        self.BoundingBoxesGroupBox.setEnabled(False)
        self.UtilitiesGroupBox.setEnabled(False)
        self.SaveImageButton.setEnabled(True)

        # Returning to default settings
        self.ChannelsComboBox.setCurrentIndex(0)
        self.ThresholdSlider.setValue(127)
        self.ThresholdSpinBox.setValue(127)
        self.DrawBoundingBoxesCheckBox.setCheckState(False)
        self.InverseBinaryCheckBox.setCheckState(False)
        self.ShowOriginalImageCheckBox.setCheckState(False)

        # Displaying the image
        pixmap = QPixmap(self.File_load_path)
        self.ImageLabel.setPixmap(QPixmap(pixmap))
    
    def UpdateImage(self, return_to_orig = False):
        '''
        Function, that updates image in the ImageLabel
        '''
        if return_to_orig:
            pixmap = QPixmap(self.File_load_path)
            self.ImageLabel.setPixmap(QPixmap(pixmap))
        else:
            if self.DrawBoundingBoxesCheckBox.isChecked():
                if self.ShowOriginalImageCheckBox.isChecked():
                    self.displayed_image = draw_annotations(self.original_bacteria_image.copy(), get_boxes(self.changed_bacteria_image.copy()),
                                                            thickness= 1, color= (255, 0, 0))
                else:
                    self.displayed_image = self.changed_bounded_bacteria_image.copy()
            else:
                if self.ShowOriginalImageCheckBox.isChecked():
                    self.displayed_image = self.original_bacteria_image.copy()
                else:
                    self.displayed_image = self.changed_bacteria_image.copy()
            height, width = self.displayed_image.shape[0], self.displayed_image.shape[1]
            if self.ShowOriginalImageCheckBox.isChecked():
                Q_displayed_image = QImage(self.displayed_image.data, width, height, QImage.Format_RGB888).rgbSwapped()
            else:
                Q_displayed_image = QImage(self.displayed_image.data, width, height, QImage.Format_Grayscale8)
            Q_displayed_image_pixmap = QPixmap.fromImage(Q_displayed_image)
            self.ImageLabel.setPixmap(Q_displayed_image_pixmap)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
