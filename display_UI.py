# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
# 
# Created by: PyQt5 UI code generator 5.15.6
#
# @Time: 2022/6/28 下午11:41
# @Author: Yang Shuo
# @File: display_UI.py
# @Software: PyCharm

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
from matplotlib import pyplot as plt
import numpy as np
import PIL.Image as Image
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import *

import os
import sys
from PyQt5 import QtWidgets, QtGui

import rubbish

class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(844, 666)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(240, 490, 131, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(240, 530, 131, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(200, 10, 431, 421))
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(570, 480, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(570, 520, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(100, 570, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(590, 570, 111, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(250, 570, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(420, 570, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(380, 490, 131, 21))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(380, 530, 131, 21))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(190, 570, 31, 21))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(350, 570, 31, 21))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_5.setGeometry(QtCore.QRect(510, 570, 31, 21))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_6.setGeometry(QtCore.QRect(710, 570, 31, 21))
        self.lineEdit_6.setObjectName("lineEdit_6")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 844, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "垃圾分类"))
        self.label.setText(_translate("MainWindow", "测试图片类别："))
        self.label_2.setText(_translate("MainWindow", "预测图片类别："))
        self.pushButton.setText(_translate("MainWindow", "选择照片"))
        self.pushButton_2.setText(_translate("MainWindow", "开始识别"))
        self.label_4.setText(_translate("MainWindow", "其他垃圾："))
        self.label_5.setText(_translate("MainWindow", "有害垃圾："))
        self.label_6.setText(_translate("MainWindow", "厨余垃圾："))
        self.label_7.setText(_translate("MainWindow", "可回收物："))
        self.pushButton.clicked.connect(self.loadFile)
        self.pushButton_2.clicked.connect(self.load_text)

    def loadFile(self):
        from PyQt5.QtGui import QPixmap
        print("load--file")
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', 'D:\\lesson-4', 'Image files(*.jpg *.gif *.png *.jpeg)')
        self.label_3.setPixmap(QPixmap(fname))
        self.label_3.setScaledContents(True)
        self.fnam = fname

    def load_text(self):
        self.lineEdit_3.setStyleSheet("background:")
        self.lineEdit_4.setStyleSheet("background:")
        self.lineEdit_5.setStyleSheet("background:")
        self.lineEdit_6.setStyleSheet("background:")
        list = rubbish.shibie(self.fnam)
        print(self.fnam)
        realtype=self.fnam.split('/')[3]
        data = str(list)
        type={'其他垃圾':0,'厨余垃圾':0,'可回收物':0,'有害垃圾':0}
        type[str(data.split()[0].split('_')[0])]=1
        print(data.split()[0].split('_')[0])
        print(type)
        # if type['其他垃圾']>0 :
        #     self.lineEdit_3.setText(str(type['其他垃圾']))
        # if type['厨余垃圾'] > 0:
        #     self.lineEdit_4.setText(str(type['厨余垃圾']))
        # if type['可回收物'] > 0:
        #     self.lineEdit_5.setText(str(type['可回收物']))
        # if type['有害垃圾'] > 0:
        #     self.lineEdit_6.setText(str(type['有害垃圾']))
        if type['其他垃圾']==1 :
            self.lineEdit_3.setStyleSheet("background:blue")
        elif type['厨余垃圾'] == 1:
            self.lineEdit_4.setStyleSheet("background:green")
        elif type['可回收物'] == 1:
            self.lineEdit_5.setStyleSheet("background:yellow")
        elif type['有害垃圾'] == 1:
            self.lineEdit_6.setStyleSheet("background:black")
        self.lineEdit_2.setText(data.split()[0])
        self.lineEdit.setText(realtype)

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()  # 创建窗体类对象MainWindow
    ui = Ui_MainWindow()  # 创建PyQT设计的窗体对象
    ui.setupUi(MainWindow)  # 初始化MainWindow窗口设置
    MainWindow.show()  # 显示窗口
    sys.exit(app.exec_())
