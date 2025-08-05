#!/usr/bin/python
# -*- coding: UTF-8 -*-
import random
import sys
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QPalette, QBrush, QPixmap
import os
import time
import PySide2
import numpy as np
from driver_detection import Face_mesh
from UI.ui_driver import Ui_Form
from pygame import mixer

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

class Driver_Monitoring_System():
    def __init__(self):
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.fm = Face_mesh()
        self.CAM_NUM = 0
        self.sound_file = "./UI/alarm.mp3"
        self.sound_play = False
        self.last_play = False
        mixer.init()
        mixer.music.load(self.sound_file)
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.count = 0
        self.fps = []
        self.aver_fps = 0

    def set_ui(self):
        self.MainWindow = QMainWindow()
        self.ui = Ui_Form()
        self.ui.setupUi(self.MainWindow)
        r = self.ui.ratio
        self.green_drinking = cv2.imread('./UI/green_drinking.png')
        self.green_drinking = cv2.resize(self.green_drinking, (int(60 * r), int(60 * r)))
        self.green_drinking = cv2.cvtColor(self.green_drinking, cv2.COLOR_BGR2RGB)
        self.green_phoning = cv2.imread('./UI/green_phoning.png')
        self.green_phoning = cv2.resize(self.green_phoning, (int(60 * r), int(60 * r)))
        self.green_phoning = cv2.cvtColor(self.green_phoning, cv2.COLOR_BGR2RGB)
        self.green_texting = cv2.imread('./UI/green_texting.png')
        self.green_texting = cv2.resize(self.green_texting, (int(60 * r), int(60 * r)))
        self.green_texting = cv2.cvtColor(self.green_texting, cv2.COLOR_BGR2RGB)
        self.red_drinking = cv2.imread('./UI/red_drinking.png')
        self.red_drinking = cv2.resize(self.red_drinking, (int(60 * r), int(60 * r)))
        self.red_drinking = cv2.cvtColor(self.red_drinking, cv2.COLOR_BGR2RGB)
        self.red_phoning = cv2.imread('./UI/red_phoning.png')
        self.red_phoning = cv2.resize(self.red_phoning, (int(60 * r), int(60 * r)))
        self.red_phoning = cv2.cvtColor(self.red_phoning, cv2.COLOR_BGR2RGB)
        self.red_texting = cv2.imread('./UI/red_texting.png')
        self.red_texting = cv2.resize(self.red_texting, (int(60 * r), int(60 * r)))
        self.red_texting = cv2.cvtColor(self.red_texting, cv2.COLOR_BGR2RGB)
        self.happy = cv2.imread('./UI/happy.jpg')
        self.happy = cv2.resize(self.happy, (int(80 * r), int(80 * r)))
        self.happy = cv2.cvtColor(self.happy, cv2.COLOR_BGR2RGB)
        self.angry = cv2.imread('./UI/angry.jpg')
        self.angry = cv2.resize(self.angry, (int(80 * r), int(80 * r)))
        self.angry = cv2.cvtColor(self.angry, cv2.COLOR_BGR2RGB)
        self.MainWindow.setWindowTitle("Driver monitoring system       AutoMan @ NTU")
        self.MainWindow.setWindowIcon(QtGui.QIcon('AutoMan.ico'))

        button_color = [self.ui.button_open_camera, self.ui.button_close]
        for i in range(2):
            button_color[i].setStyleSheet("QPushButton{color:black}"
                                          "QPushButton:hover{color:red}"
                                          "QPushButton{background-color:rgb(246,197,18)}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:6px}"
                                          "QPushButton{padding:2px 2px}")

        self.ui.button_open_camera.setMinimumHeight(int(20 * r))
        self.ui.button_close.setMinimumHeight(int(20 * r))

        self.xdata = list(range(len(self.fm.yawn_interval)))
        self.ydata = [1/(interval + 0.00000001) for interval in self.fm.yawn_interval]
        self.x = list(range(len(self.fm.blink_interval)))
        self.y = [1/(interval + 0.00000001) for interval in self.fm.blink_interval]

    def update_plot(self):
        self.xdata = list(range(len(self.fm.yawn_interval)))
        self.ydata = [1 / (interval + 0.00000001) for interval in self.fm.yawn_interval]
        self.ui.canvas1.axes.cla()
        self.ui.canvas1.axes.plot(self.xdata, self.ydata, 'orange', alpha=1)
        self.ui.canvas1.axes.set_ylim(bottom=0, top=40)
        self.ui.canvas1.axes.fill_between(self.xdata, self.ydata, y2=0, color='orange', alpha=1, label='area')
        self.ui.canvas1.draw()

        self.x = list(range(len(self.fm.blink_interval)))
        self.y = [1 / (interval + 0.00000001) for interval in self.fm.blink_interval]
        self.ui.canvas2.axes.cla()
        self.ui.canvas2.axes.plot(self.x, self.y, 'b', alpha=1)
        self.ui.canvas2.axes.set_ylim(bottom=0, top=3000000)
        self.ui.canvas2.axes.fill_between(self.x, self.y, y2=0, color='b', alpha=1, label='area')
        self.ui.canvas2.draw()


    def slot_init(self):
        self.ui.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.ui.button_close.clicked.connect(QCoreApplication.instance().quit)


    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please check the connection of camera!",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)
                self.ui.button_open_camera.setText(u'Close Camera')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.ui.camera_show.clear()
            self.ui.button_open_camera.setText(u'Open Camera')

    def show_camera(self):
        r = self.ui.ratio
        start_time = time.time()
        flag, self.image = self.cap.read()
        if flag == False:
            self.cap.open(self.CAM_NUM)
        else:
            output = self.fm.get_3D_face_mesh(self.image)
            show = cv2.resize(self.image, (int(640 * r), int(480 * r)))
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            cv2.rectangle(show, (0, 0), (int(640 * r), int(40 * r)), color=(180, 180, 180), thickness=-1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(show, str("FPS: %.2f" % (self.aver_fps)), (int(10 * r), int(30 * r)), font, 0.6 * r, (0, 0, 255), int(2 * r))

            annotated_image = cv2.cvtColor(output[1], cv2.COLOR_BGR2RGB)
            annotated_image = cv2.resize(annotated_image, (int(280 * r), int(200 * r)))
            left_eye = cv2.cvtColor(output[2], cv2.COLOR_BGR2RGB)
            left_eye = cv2.resize(left_eye, (int(120 * r), int(60 * r)))
            right_eye = cv2.cvtColor(output[3], cv2.COLOR_BGR2RGB)
            right_eye = cv2.resize(right_eye, (int(120 * r), int(60 * r)))
            mouth = cv2.cvtColor(output[4], cv2.COLOR_BGR2RGB)
            mouth = cv2.resize(mouth, (int(120 * r), int(80 * r)))

            show[int(50 * r):int(110 * r), int(530 * r):int(590 * r)] = self.green_drinking
            show[int(120 * r):int(180 * r), int(530 * r):int(590 * r)] = self.green_phoning
            show[int(190 * r):int(250 * r), int(530 * r):int(590 * r)] = self.green_texting
            showImage1 = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            showImage2 = QtGui.QImage(annotated_image.data, annotated_image.shape[1], annotated_image.shape[0], QtGui.QImage.Format_RGB888)
            showImage3 = QtGui.QImage(left_eye.data, left_eye.shape[1], left_eye.shape[0], QtGui.QImage.Format_RGB888)
            showImage4 = QtGui.QImage(right_eye.data, right_eye.shape[1], right_eye.shape[0], QtGui.QImage.Format_RGB888)
            showImage5 = QtGui.QImage(mouth.data, mouth.shape[1], mouth.shape[0], QtGui.QImage.Format_RGB888)

            head_roll = np.sum(self.fm.angle_roll) / len(self.fm.angle_roll)
            head_yaw = np.sum(self.fm.angle_yaw) / len(self.fm.angle_yaw)
            head_pitch = np.sum(self.fm.angle_pitch) / len(self.fm.angle_pitch)
            gaze_pitch = np.sum(self.fm.gaze_pitch) / len(self.fm.gaze_pitch)
            gaze_yawn = np.sum(self.fm.gaze_yaw) / len(self.fm.gaze_yaw)
            self.ui.gaze_pitch.setText("%.1f" % (gaze_pitch))
            self.ui.gaze_yaw.setText("%.1f" % (gaze_yawn))
            expression_score = 0
            behavior_score = 0

            head_yaw_score, head_pitch_score = 0, 0
            gaze_yawn_score, gaze_pitch_score = 0, 0
            if head_yaw > 0:
                head_yaw_score = 60 / 40 * head_yaw
            else:
                head_yaw_score = -head_yaw
            if head_pitch > 0:
                head_pitch_score = 2 * head_pitch
            else:
                head_pitch_score = - head_pitch * 2
            if gaze_yawn > 0:
                gaze_yawn_score = 60 / 50 * gaze_yawn
            else:
                gaze_yawn_score = - gaze_yawn
            if gaze_pitch > 0:
                gaze_pitch_score = 60/40 * gaze_pitch
            else:
                gaze_pitch_score = - gaze_pitch * 3

            dis_score = np.max([head_yaw_score, head_pitch_score, gaze_yawn_score, gaze_pitch_score])
            if self.fm.eye_close or self.fm.yawning:
                distraction_score = dis_score / 10
            elif dis_score < 52:
                distraction_score = dis_score / (53-dis_score)
            else:
                distraction_score = dis_score

            drowsiness_score1 = self.fm.blink_duration / 100 * 33 + 6 * (len(self.fm.blink_frequency) - 2)
            drowsiness_score2 = self.fm.yawn_duration / 100 * 33 + 20 * (len(self.fm.yawn_frequency) - 1)
            drowsiness_score = np.min([drowsiness_score1, drowsiness_score2, 0])

            if self.fm.abnormal_behaviour in ['drinking', 'answering the phone', 'texting with phone']:
                distraction_score = dis_score / 10
                drowsiness_score = drowsiness_score / 10
                behavior_score = self.fm.behaviour_score
            if self.fm.expression in ['Happy', 'Angry']:
                distraction_score = dis_score / 10
                drowsiness_score = drowsiness_score / 10
                expression_score = self.fm.expression_score

            if self.fm.abnormal_behaviour in ['drinking', 'answering the phone', 'texting with phone']:
                cv2.putText(show, "Abnormal detected!!!", (int(230 * r), int(26 * r)), font, 0.6 * r, (255, 0, 0), int(2 * r))
            else:
                cv2.putText(show, "---Normal driving---", (int(230 * r), int(26 * r)), font, 0.6 * r, (0, 255, 0), int(2 * r))

            if self.fm.abnormal_behaviour == 'drinking':
                show[int(50 * r):int(110 * r), int(530 * r):int(590 * r)] = self.red_drinking
            elif self.fm.abnormal_behaviour == 'answering the phone':
                show[int(120 * r):int(180 * r), int(530 * r):int(590 * r)] = self.red_phoning
            elif self.fm.abnormal_behaviour == 'texting with phone':
                show[int(190 * r):int(250 * r), int(530 * r):int(590 * r)] = self.red_texting

            # if self.fm.expression == 'Happy':
            #     show[int(50 * r):int(130 * r), int(10 * r):int(90 * r)] = self.happy
            # elif self.fm.expression == 'Angry':
            #     show[int(50 * r):int(130 * r), int(10 * r):int(90 * r)] = self.angry

            self.ui.camera_show.setPixmap(QtGui.QPixmap.fromImage(showImage1))
            self.ui.face_mesh.setPixmap(QtGui.QPixmap.fromImage(showImage2))
            self.ui.left_eye.setPixmap(QtGui.QPixmap.fromImage(showImage3))
            self.ui.right_eye.setPixmap(QtGui.QPixmap.fromImage(showImage4))
            self.ui.mouth.setPixmap(QtGui.QPixmap.fromImage(showImage5))

            if self.fm.blink_duration < 200:
                self.ui.blink_duration.setStyleSheet("QProgressBar::chunk"
                                                     "{"
                                                     "background-color:green;"
                                                     "text-align:center;"
                                                     "}")
            else:
                self.ui.blink_duration.setStyleSheet("QProgressBar::chunk"
                                                     "{"
                                                     "background-color:red;"
                                                     "}")
            if distraction_score < 60:
                self.ui.distraction_detection.setStyleSheet("QProgressBar::chunk"
                                                     "{"
                                                     "background-color:green;"
                                                     "text-align:center;"
                                                     "}")
            else:
                self.ui.distraction_detection.setStyleSheet("QProgressBar::chunk"
                                                     "{"
                                                     "background-color:red;"
                                                     "}")
            if drowsiness_score < 60:
                self.ui.drowsiness_detection.setStyleSheet("QProgressBar::chunk"
                                                     "{"
                                                     "background-color:green;"
                                                     "text-align:center;"
                                                     "}")
            else:
                self.ui.drowsiness_detection.setStyleSheet("QProgressBar::chunk"
                                                     "{"
                                                     "background-color:red;"
                                                     "}")
            if expression_score < 60:
                self.ui.emotion_detection.setStyleSheet("QProgressBar::chunk"
                                                     "{"
                                                     "background-color:green;"
                                                     "text-align:center;"
                                                     "}")
            else:
                self.ui.emotion_detection.setStyleSheet("QProgressBar::chunk"
                                                     "{"
                                                     "background-color:red;"
                                                     "}")
            if behavior_score < 60:
                self.ui.behavior_detection.setStyleSheet("QProgressBar::chunk"
                                                     "{"
                                                     "background-color:green;"
                                                     "text-align:center;"
                                                     "}")
            else:
                self.ui.behavior_detection.setStyleSheet("QProgressBar::chunk"
                                                     "{"
                                                     "background-color:red;"
                                                     "}")

            self.ui.blink_duration.setValue(float(self.fm.blink_duration))
            self.ui.drowsiness_detection.setValue(float(drowsiness_score))
            self.ui.distraction_detection.setValue(float(distraction_score))
            self.ui.emotion_detection.setValue(0*float(expression_score))
            self.ui.behavior_detection.setValue(float(behavior_score))

            if (drowsiness_score > 60 or distraction_score > 60 or self.fm.abnormal_behaviour == 'drinking' or
                    self.fm.abnormal_behaviour == 'answering the phone' or self.fm.abnormal_behaviour == 'texting with phone' or
                    expression_score > 60 or behavior_score > 60):
                self.sound_play = True
            else:
                self.sound_play = False

            if self.sound_play != self.last_play:
                if self.sound_play == True:
                    mixer.music.play(-1)
                else:
                    mixer.music.stop()

            self.last_play = self.sound_play

            end_time = time.time()
            process_time = end_time - start_time
            self.fps.append(1/process_time)
            if len(self.fps) > 100:
                self.fps.pop(0)
            self.aver_fps = np.sum(self.fps) / len(self.fps)
            self.update_plot()


    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"Close", u"Do you want to close?")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'OK')
        cacel.setText(u'Cancel')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()


if __name__ == "__main__":
    App = QApplication(sys.argv)
    driver = Driver_Monitoring_System()
    driver.MainWindow.show()
    sys.exit(App.exec_())