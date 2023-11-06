import sys
sys.path.append('../Bacteria_finder')
sys.path.append('../Bacteria_finder/GUI application')
sys.path.append('../Bacteria_finder/Bacteria_finder_core')

from cv2 import IMREAD_UNCHANGED, imdecode, imencode
from numpy import fromfile, uint8
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap

from Bacteria_finder_core.segmentor import Bacteria_segmentor


class Ui_MainWindow(object):

    def __init__(self):
        self.segmentor = Bacteria_segmentor("omnipose")
        self.original_bacteria_image = None
        self.shown_bacteria_image = None
        self.segmented_bacteria_image = None
        self.classified_bacteria_image = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1024, 804)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        MainWindow.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        MainWindow.setFont(font)
        MainWindow.setWindowOpacity(3.0)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.ImageGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ImageGroupBox.sizePolicy().hasHeightForWidth())
        self.ImageGroupBox.setSizePolicy(sizePolicy)
        self.ImageGroupBox.setMaximumSize(QtCore.QSize(900, 900))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.ImageGroupBox.setFont(font)
        self.ImageGroupBox.setObjectName("ImageGroupBox")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.ImageGroupBox)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.ImageLabel = QtWidgets.QLabel(self.ImageGroupBox)
        self.ImageLabel.setText("")
        self.ImageLabel.setPixmap(QPixmap("GUI application/blank.png"))
        self.ImageLabel.setScaledContents(True)
        self.ImageLabel.setObjectName("ImageLabel")
        self.horizontalLayout.addWidget(self.ImageLabel)
        self.horizontalLayout_2.addWidget(self.ImageGroupBox)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.ButtonsLayout = QtWidgets.QVBoxLayout()
        self.ButtonsLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.ButtonsLayout.setObjectName("ButtonsLayout")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.ButtonsLayout.addItem(spacerItem1)
        self.LoadImageButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.LoadImageButton.setFont(font)
        self.LoadImageButton.setObjectName("LoadImageButton")
        self.ButtonsLayout.addWidget(self.LoadImageButton)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.ButtonsLayout.addItem(spacerItem2)
        self.SegmentImageButton = QtWidgets.QPushButton(self.centralwidget)
        self.SegmentImageButton.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.SegmentImageButton.setFont(font)
        self.SegmentImageButton.setObjectName("SegmentImageButton")
        self.ButtonsLayout.addWidget(self.SegmentImageButton)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.ButtonsLayout.addItem(spacerItem3)
        self.ClassifyImageButton = QtWidgets.QPushButton(self.centralwidget)
        self.ClassifyImageButton.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.ClassifyImageButton.setFont(font)
        self.ClassifyImageButton.setObjectName("ClassifyImageButton")
        self.ButtonsLayout.addWidget(self.ClassifyImageButton)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.ButtonsLayout.addItem(spacerItem4)
        self.SaveImageButton = QtWidgets.QPushButton(self.centralwidget)
        self.SaveImageButton.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        self.SaveImageButton.setFont(font)
        self.SaveImageButton.setObjectName("SaveImageButton")
        self.ButtonsLayout.addWidget(self.SaveImageButton)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.ButtonsLayout.addItem(spacerItem5)
        self.CountertextBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.CountertextBrowser.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.CountertextBrowser.sizePolicy().hasHeightForWidth())
        self.CountertextBrowser.setSizePolicy(sizePolicy)
        self.CountertextBrowser.setMinimumSize(QtCore.QSize(0, 30))
        self.CountertextBrowser.setMaximumSize(QtCore.QSize(16777215, 110))
        self.CountertextBrowser.setObjectName("CountertextBrowser")
        self.ButtonsLayout.addWidget(self.CountertextBrowser)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.ButtonsLayout.addItem(spacerItem6)
        self.ShowObjectsgroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.ShowObjectsgroupBox.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ShowObjectsgroupBox.sizePolicy().hasHeightForWidth())
        self.ShowObjectsgroupBox.setSizePolicy(sizePolicy)
        self.ShowObjectsgroupBox.setMinimumSize(QtCore.QSize(0, 170))
        self.ShowObjectsgroupBox.setObjectName("ShowObjectsgroupBox")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.ShowObjectsgroupBox)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 20, 160, 126))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.ShowObjectsverticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.ShowObjectsverticalLayout.setContentsMargins(0, 0, 0, 0)
        self.ShowObjectsverticalLayout.setObjectName("ShowObjectsverticalLayout")
        self.AllradioButton = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.AllradioButton.setObjectName("AllradioButton")
        self.ShowObjectsverticalLayout.addWidget(self.AllradioButton)
        self.BacillusradioButton = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.BacillusradioButton.setObjectName("BacillusradioButton")
        self.ShowObjectsverticalLayout.addWidget(self.BacillusradioButton)
        self.CoccusradioButton = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.CoccusradioButton.setObjectName("CoccusradioButton")
        self.ShowObjectsverticalLayout.addWidget(self.CoccusradioButton)
        self.GroupsradioButton = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.GroupsradioButton.setObjectName("GroupsradioButton")
        self.ShowObjectsverticalLayout.addWidget(self.GroupsradioButton)
        self.MiscradioButton = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.MiscradioButton.setObjectName("MiscradioButton")
        self.ShowObjectsverticalLayout.addWidget(self.MiscradioButton)
        self.ButtonsLayout.addWidget(self.ShowObjectsgroupBox)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.ButtonsLayout.addItem(spacerItem7)
        self.horizontalLayout_2.addLayout(self.ButtonsLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1024, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # All the connections are below
        # Load and Save
        self.LoadImageButton.clicked.connect(self.LoadImageButtonPushed)
        self.SaveImageButton.clicked.connect(self.SaveImageButtonPushed)

        # Segment image
        self.SegmentImageButton.clicked.connect(self.SegmentImageButtonPushed)

        # Classify image
        self.ClassifyImageButton.clicked.connect(self.ClassifyImageButtonPushed)

        # Change displayed objects by radiobuttons
        self.AllradioButton.toggled.connect(self.RadioButtonToggled)
        self.BacillusradioButton.toggled.connect(self.RadioButtonToggled)
        self.CoccusradioButton.toggled.connect(self.RadioButtonToggled)
        self.GroupsradioButton.toggled.connect(self.RadioButtonToggled)
        self.MiscradioButton.toggled.connect(self.RadioButtonToggled)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Bacteria finder"))
        self.ImageGroupBox.setTitle(_translate("MainWindow", "Image"))
        self.LoadImageButton.setText(_translate("MainWindow", "Load image"))
        self.SegmentImageButton.setText(_translate("MainWindow", "Segment"))
        self.ClassifyImageButton.setText(_translate("MainWindow", "Classify"))
        self.SaveImageButton.setText(_translate("MainWindow", "Save image"))
        self.CountertextBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Sans Serif\'; font-size:9.75pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Counter:</p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Objects = 0</p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Bacillus = 0</p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Coccus = 0</p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Groups = 0</p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Misc = 0</p></body></html>"))
        self.ShowObjectsgroupBox.setTitle(_translate("MainWindow", "Show objects"))
        self.AllradioButton.setText(_translate("MainWindow", "All"))
        self.BacillusradioButton.setText(_translate("MainWindow", "Bacillus"))
        self.CoccusradioButton.setText(_translate("MainWindow", "Coccus"))
        self.GroupsradioButton.setText(_translate("MainWindow", "Groups"))
        self.MiscradioButton.setText(_translate("MainWindow", "Misc"))

    def RadioButtonToggled(self):
        '''
        Updates shown image by which radio button is pressed
        '''
        if self.AllradioButton.isChecked():
            self.shown_bacteria_image = self.classified_bacteria_image.copy()
        elif self.BacillusradioButton.isChecked():
            self.shown_bacteria_image = self.segmentor.image_out_bacili.copy()
        elif self.CoccusradioButton.isChecked():
            self.shown_bacteria_image = self.segmentor.image_out_cocci.copy()
        elif self.GroupsradioButton.isChecked():
            self.shown_bacteria_image = self.segmentor.image_out_grouped.copy()
        elif self.MiscradioButton.isChecked():
            self.shown_bacteria_image = self.segmentor.image_out_misc.copy()
        self.UpdateImage()
    
    def UpdateCounter(self):
        '''
        Updates on screen counter
        '''
        self.CountertextBrowser.setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                            "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                            "p, li { white-space: pre-wrap; }\n"
                                            "</style></head><body style=\" font-family:\'MS Sans Serif\'; font-size:9.75pt; font-weight:400; font-style:normal;\">\n"
                                            "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Counter:</p>\n"
                                            f"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Objects = {self.segmentor.objects_num['Objects']}</p>\n"
                                            f"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Bacillus = {self.segmentor.objects_num['Bacillus']}</p>\n"
                                            f"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Coccus = {self.segmentor.objects_num['Coccus']}</p>\n"
                                            f"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Groups = {self.segmentor.objects_num['Groups']}</p>\n"
                                            f"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Misc = {self.segmentor.objects_num['Misc']}</p></body></html>")
    
    def ClassifyImageButtonPushed(self):
        '''
        Function that classifies the image and displays the result
        '''

        if self.classified_bacteria_image is None:
            self.classified_bacteria_image = self.segmentor.pipeline(self.original_bacteria_image, "classify")
            self.ShowObjectsgroupBox.setEnabled(True)
        self.AllradioButton.setChecked(True)
        self.UpdateCounter()
        self.shown_bacteria_image = self.classified_bacteria_image.copy()
        self.UpdateImage()
    
    def SegmentImageButtonPushed(self):
        '''
        Function that segments the image and displays the result
        '''

        if self.segmented_bacteria_image is None:
            self.segmented_bacteria_image = self.segmentor.pipeline(self.original_bacteria_image, "segment")
            self.CountertextBrowser.setEnabled(True)
            self.ClassifyImageButton.setEnabled(True)
        if self.ShowObjectsgroupBox.isEnabled():
            self.AllradioButton.setChecked(True)
        self.UpdateCounter()
        self.shown_bacteria_image = self.segmented_bacteria_image.copy()
        self.UpdateImage()
    
    def SaveImageButtonPushed(self):
        '''
        Function that opens save dialog window and saves the shown image
        '''
        # Getting the path for the image
        self.File_save_name = QtWidgets.QFileDialog.getSaveFileName(self.ImageLabel, 'Save file', '', "Image files (*.png *.jpg *.bmp *.gif)")
        self.File_save_path = self.File_save_name[0]

        # Check if the user didn't choose the path
        if self.File_save_path == '':
            return

        # Saving in choisen path
        _, buffed_image = imencode("." + self.File_save_path.split('/')[-1].split('.')[-1], self.shown_bacteria_image.copy())
        buffed_image.tofile(self.File_save_path)

    def LoadImageButtonPushed(self):
        '''
        Function that loads the path to an image
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
        # Deleting previous results
        self.segmentor = Bacteria_segmentor("omnipose")
        self.UpdateCounter()
        self.original_bacteria_image = None
        self.shown_bacteria_image = None
        self.segmented_bacteria_image = None
        self.classified_bacteria_image = None

        # Saving the image for future use in cv2
        self.shown_bacteria_image = imdecode(fromfile(self.File_load_path, dtype=uint8), IMREAD_UNCHANGED)
        self.original_bacteria_image = self.shown_bacteria_image.copy()

        # Enabling and disabling widgets
        self.SegmentImageButton.setEnabled(True)
        self.ClassifyImageButton.setEnabled(False)
        self.SaveImageButton.setEnabled(True)
        self.LoadImageButton.setEnabled(True)
        self.ShowObjectsgroupBox.setEnabled(False)
        self.CountertextBrowser.setEnabled(False)

        # Displaying the image
        pixmap = QPixmap(self.File_load_path)
        self.ImageGroupBox.setMaximumSize(QtCore.QSize(int(900*self.shown_bacteria_image.shape[1]/self.shown_bacteria_image.shape[0]), 900))
        self.ImageLabel.setPixmap(QPixmap(pixmap))
    
    def UpdateImage(self, return_to_orig = False):
        '''
        Function, that updates image in the ImageLabel
        '''
            
        height, width = self.shown_bacteria_image.shape[0], self.shown_bacteria_image.shape[1]
        Q_displayed_image = QImage(self.shown_bacteria_image.data, width, height, width * 3, QImage.Format_RGB888).rgbSwapped()
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