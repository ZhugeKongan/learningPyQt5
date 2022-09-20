#!/usr/bin/env python3
###################################################################################################
#
# Copyright (C) 2018-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""Thread to capture image from camera continiously
"""
import time

from PyQt5.QtCore import (QThread, pyqtSignal, Qt)
from PyQt5.QtGui import QImage

from image_utils import cvt_img_to_qimage

'''QT 多线程的使用主要是通过 QThread 来实现。有两种方法：一种是创建一个继承自QThread的类并重写它的run()方法；
另一种是，创建类，实例化对象并转换为线程对象。
'''
class Thread(QThread):
    """
    Thread to capture image from camera
    """
    # 自定义信号
    change_pixmap = pyqtSignal(QImage)

    # 构造函数，接受参数
    def __init__(self, parent=None, camera=None, frame_rate=25):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        self.emit_period = 1.0 / frame_rate

    # 重写run()方法
    def run(self):
        """Runs camera capture"""
        prev = time.time()
        while True:
            now = time.time()
            # 图像提取框
            rval, frame = self.camera.get_frame()
            # 转换图片格式
            if rval:
                convert_qt_format = cvt_img_to_qimage(frame)
                qt_img = convert_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                if (now - prev) >= self.emit_period:
                    # 发射信号，执行它所连接的函数
                    self.change_pixmap.emit(qt_img)
                    prev = now
