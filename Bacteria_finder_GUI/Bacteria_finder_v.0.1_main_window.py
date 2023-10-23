from sys import path

path.append('../Bacteria_finder')
path.append('../Bacteria_finder/Bacteria_finder_GUI')
path.append('../Bacteria_finder/Bacteria_finder_core')

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
        self.ImageLabel.setPixmap(QPixmap("Bacteria_finder_GUI/blank.png"))
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

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Bacteria finder"))
        self.ImageGroupBox.setTitle(_translate("MainWindow", "Image"))
        self.LoadImageButton.setText(_translate("MainWindow", "Load image"))
        self.SegmentImageButton.setText(_translate("MainWindow", "Segment"))
        self.ClassifyImageButton.setText(_translate("MainWindow", "Classify"))
        self.SaveImageButton.setText(_translate("MainWindow", "Save image"))

    def ClassifyImageButtonPushed(self):
        '''
        Function that classifies the image and displays the result
        '''

        if self.classified_bacteria_image is None:
            self.classified_bacteria_image = self.segmentor.pipeline(self.original_bacteria_image, "all")
        self.shown_bacteria_image = self.classified_bacteria_image.copy()
        self.UpdateImage()
    
    def SegmentImageButtonPushed(self):
        '''
        Function that segments the image and displays the result
        '''

        if self.segmented_bacteria_image is None:
            self.segmented_bacteria_image = self.segmentor.pipeline(self.original_bacteria_image, "segment")
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
        self.original_bacteria_image = None
        self.shown_bacteria_image = None
        self.segmented_bacteria_image = None
        self.classified_bacteria_image = None

        # Saving the image for future use in cv2
        self.shown_bacteria_image = imdecode(fromfile(self.File_load_path, dtype=uint8), IMREAD_UNCHANGED)
        self.original_bacteria_image = self.shown_bacteria_image.copy()

        # Enabling and disabling widgets
        self.SegmentImageButton.setEnabled(True)
        self.ClassifyImageButton.setEnabled(True)
        self.SaveImageButton.setEnabled(True)
        self.LoadImageButton.setEnabled(True)

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