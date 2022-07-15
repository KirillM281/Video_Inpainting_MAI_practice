# -*- coding: utf-8 -*-
import cv2
from PIL import Image
from PIL.ImageQt import ImageQt
import numpy as np
import importlib
import sys
import time
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import torch

from core.utils import to_tensors

from PyQt5.QtCore import QThread, pyqtSignal, QUrl, Qt, QRectF, QPointF, QFile, QIODevice
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon, QPainter, QPixmap, QPen
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget

import design

ref_length = 10  # ref_step
num_ref = -1
neighbor_stride = 5
default_fps = 24

def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref // 2))
        end_idx = min(length, f + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index


# read frame-wise masks
def read_mask(mpath, size):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for mp in mnames:
        m = Image.open(os.path.join(mpath, mp))
        m = m.resize(size, Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m,
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                       iterations=4)
        masks.append(Image.fromarray(m * 255))
    return masks

def clone_mask(mpath, size, maskcnt):
    masks = []
    for _ in range(maskcnt):
        m = Image.open(mpath)
        m = m.resize(size, Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m,
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                       iterations=4)
        masks.append(Image.fromarray(m * 255))
    return masks

#  read frames from video
def read_frame_from_videos(video, use_mp4):
    vname = video
    frames = []
    if use_mp4:
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        count = 0
        while success:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image)
            success, image = vidcap.read()
            count += 1
    else:
        lst = os.listdir(vname)
        lst.sort()
        fr_lst = [vname + '/' + name for name in lst]
        for fr in fr_lst:
            image = cv2.imread(fr)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image)
    return frames


# resize frames
def resize_frames(frames, size=None):
    if size is not None:
        frames = [f.resize(size) for f in frames]
    else:
        size = frames[0].size
    return frames, size

def get_frame_from_video(video, use_mp4):
    vname = video
    if use_mp4:
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        success, image = vidcap.read()
        if success:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return image

class External(QThread):
    countChanged = pyqtSignal(int)
    maxValPass = pyqtSignal(int)
    endSignal = pyqtSignal(int)
    def __init__(self, model, video, ckpt, mask, savepass, wtm = False):
        super(QThread, self).__init__()
        self.model = model
        self.video = video
        self.ckpt = ckpt
        self.mask = mask
        self.set_size = None
        self.width = None
        self.height = None
        self.savepass = savepass
        self.wtm = wtm
    def run(self):
        # set up models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model == "e2fgvi":
            size = (432, 240)
        elif self.set_size:
            size = (self.width, self.height)
        else:
            size = None
        net = importlib.import_module('model.' + self.model)
        self.model = net.InpaintGenerator().to(device)
        data = torch.load(self.ckpt, map_location=device)
        self.model.load_state_dict(data)
        print(f'Loading model from: {self.ckpt}')
        self.model.eval()
        # prepare datset
        use_mp4 = True if self.video.endswith('.mp4') else False
        print(
            f'Loading videos and masks from: {self.video} | INPUT MP4 format: {use_mp4}'
        )
        frames = read_frame_from_videos(self.video, use_mp4)
        frames, size = resize_frames(frames, size)
        h, w = size[1], size[0]
        video_length = len(frames)
        imgs = to_tensors()(frames).unsqueeze(0) * 2 - 1
        frames = [np.array(f).astype(np.uint8) for f in frames]
        if(self.wtm):
            masks = clone_mask(self.mask, size, video_length)
        else:
            masks = read_mask(self.mask, size)
        binary_masks = [
            np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks
        ]
        masks = to_tensors()(masks).unsqueeze(0)
        imgs, masks = imgs.to(device), masks.to(device)
        comp_frames = [None] * video_length
        # completing holes by e2fgvi
        print(f'Start test...')
        self.maxValPass.emit(video_length/neighbor_stride)
        prval = 0
        for f in tqdm(range(0, video_length, neighbor_stride)):
            self.countChanged.emit(prval)
            prval += 1
            neighbor_ids = [
                i for i in range(max(0, f - neighbor_stride),
                                 min(video_length, f + neighbor_stride + 1))
            ]
            ref_ids = get_ref_index(f, neighbor_ids, video_length)
            selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
            with torch.no_grad():
                masked_imgs = selected_imgs * (1 - selected_masks)
                mod_size_h = 60
                mod_size_w = 108
                h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
                masked_imgs = torch.cat(
                    [masked_imgs, torch.flip(masked_imgs, [3])],
                    3)[:, :, :, :h + h_pad, :]
                masked_imgs = torch.cat(
                    [masked_imgs, torch.flip(masked_imgs, [4])],
                    4)[:, :, :, :, :w + w_pad]
                pred_imgs, _ = self.model(masked_imgs, len(neighbor_ids))
                pred_imgs = pred_imgs[:, :, :h, :w]
                pred_imgs = (pred_imgs + 1) / 2
                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_imgs[i]).astype(
                        np.uint8) * binary_masks[idx] + frames[idx] * (
                            1 - binary_masks[idx])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(
                            np.float32) * 0.5 + img.astype(np.float32) * 0.5

        # saving videos
        print('Saving videos...')
        save_dir_name = self.savepass
        ext_name = '_results.mp4'
        save_base_name = self.video.split('/')[-1]
        save_name = save_base_name.replace(
            '.mp4', ext_name) if use_mp4 else save_base_name + ext_name
        if not os.path.exists(save_dir_name):
            os.makedirs(save_dir_name)
        save_path = os.path.join(save_dir_name, save_name)
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 default_fps, size)
        for f in range(video_length):
            comp = comp_frames[f].astype(np.uint8)
            writer.write(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
        writer.release()
        print(f'Finish test! The result video is saved in: {save_path}.')
        self.endSignal.emit(0)
        # show results

class ExampleApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super(ExampleApp, self).__init__()
        self.setupUi(self)
        self.PenSize = 5
        self.stackedWidget.setCurrentIndex(0)
        self.pushButton.clicked.connect(self.simple_regime)
        self.pushButton_2.clicked.connect(self.watermark_regime)
        self.pushButton_3.clicked.connect(self.exit_app)
        self.VideoChoose.clicked.connect(self.video_choose)
        self.MaskChoose.clicked.connect(self.mask_choose)
        self.SaveChoose.clicked.connect(self.save_choose)
        self.Start.clicked.connect(self.start_process)
        self.VideoChoose2.clicked.connect(self.video_choose2)
        self.VideoPlay.setIcon(QIcon('play.png'))
        self.VideoPlay.setEnabled(False)
        self.VideoPlay.clicked.connect(self.play)
        self.horizontalSlider.sliderMoved.connect(self.setPosition)
        self.QualityChooseGroup = QtWidgets.QButtonGroup()
        self.QualityChooseGroup.addButton(self.Low_quality)
        self.QualityChooseGroup.addButton(self.Any_quality)
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
        self.mediaPlayer.setVideoOutput(self.widget)
        self.BackButton2.clicked.connect(self.back2)
        self.BackButton.clicked.connect(self.back2)
        self.NextButton.clicked.connect(self.next)
        self.BackButton3.clicked.connect(self.back3)
        self.MaskChoose2.clicked.connect(self.mask_choose2)
        self.NextButton2.clicked.connect(self.next2)
        self.BackButton4.clicked.connect(self.back4)
        self.QualityChooseGroup2 = QtWidgets.QButtonGroup()
        self.QualityChooseGroup2.addButton(self.LowQuality)
        self.QualityChooseGroup2.addButton(self.AnyQuality)
        self.SaveChoose2.clicked.connect(self.save_choose2)
        self.Start2.clicked.connect(self.start_process2)
        self.coords = QPointF()
        self.NextMaskReady.clicked.connect(self.to_mask_ready)
        self.BackButton3_2.clicked.connect(self.back5)
        self.comboBox.currentIndexChanged.connect(self.change_pen_size)
        self.NextButton2_2.clicked.connect(self.next3)
        self.MaskReadyChoose.clicked.connect(self.mask_choose3)

    def next3(self):
        self.IfMaskCreated = False
        self.stackedWidget.setCurrentIndex(4)

    def change_pen_size(self, index):
        if index == 1:
            self.PenSize = 5
        elif index == 2:
            self.PenSize = 6
        elif index == 3:
            self.PenSize = 7
        elif index == 4:
            self.PenSize = 8
        elif index == 5:
            self.PenSize = 9
        elif index == 6:
            self.PenSize = 10

    def back5(self):
        self.stackedWidget.setCurrentIndex(2)

    def to_mask_ready(self):
        self.stackedWidget.setCurrentIndex(6)
    
    def back4(self):
        self.stackedWidget.setCurrentIndex(3)

    def next2(self):
        self.mask_pixmap.save("tmp_mask.png")
        self.IfMaskCreated = True
        self.stackedWidget.setCurrentIndex(4)

    def start_process2(self):
        if self.LowQuality.isChecked():
            self.stackedWidget.setCurrentIndex(5)
            if self.IfMaskCreated:
                self.calc = External("e2fgvi", self.VideoName.text(), "release_model/E2FGVI-CVPR22.pth", "tmp_mask.png", self.SavePath2.toPlainText(), True)
            else:
                self.calc = External("e2fgvi", self.VideoName.text(), "release_model/E2FGVI-CVPR22.pth", self.MaskReadyPath.text(), self.SavePath2.toPlainText(), True)
            self.calc.countChanged.connect(self.onCountChanged)
            self.calc.maxValPass.connect(self.onMaxVal)
            self.calc.endSignal.connect(self.endProcess)
            self.calc.start()
        if self.AnyQuality.isChecked():
            self.stackedWidget.setCurrentIndex(5)
            if self.IfMaskCreated:
                self.calc = External("e2fgvi_hq", self.VideoName.text(), "release_model/E2FGVI-HQ-CVPR22.pth", "tmp_mask.png", self.SavePath2.toPlainText(), True)
            else:
                self.calc = External("e2fgvi", self.VideoName.text(), "release_model/E2FGVI-CVPR22.pth", self.MaskReadyPath.text(), self.SavePath2.toPlainText(), True)
            self.calc.countChanged.connect(self.onCountChanged)
            self.calc.maxValPass.connect(self.onMaxVal)
            self.calc.endSignal.connect(self.endProcess)
            self.calc.start()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.main_painter = QPainter(self.PhotoShow.pixmap())
            self.mask_painter = QPainter(self.mask_pixmap)
            self.main_painter.setPen(QPen(Qt.blue, self.PenSize, Qt.SolidLine))
            self.mask_painter.setPen(QPen(Qt.blue, self.PenSize, Qt.SolidLine))
            tmp_pos = self.PhotoShow.mapFromParent(event.pos())
            xoffset = (self.PhotoShow.width() - self.PhotoShow.pixmap().width()) / 2
            yoffset = (self.PhotoShow.height() - self.PhotoShow.pixmap().height()) / 2
            self.main_painter.drawPoint(tmp_pos.x() - xoffset - 9, tmp_pos.y() - yoffset - 11)
            self.mask_painter.drawPoint(tmp_pos.x() - xoffset - 9, tmp_pos.y() - yoffset - 11)
            self.main_painter.end()
            self.mask_painter.end()
            self.update()

    def mask_choose2(self):
        self.main_pixmap = QPixmap.fromImage(ImageQt(self.premask_frame))
        self.main_pixmap = self.main_pixmap.scaledToHeight(self.PhotoShow.height())
        self.PhotoShow.setPixmap(self.main_pixmap)
        self.mask_pixmap = QPixmap(self.main_pixmap.width(), self.main_pixmap.height())
        self.mask_pixmap.fill(Qt.black)

    def mask_choose3(self):
        MaskDirectory = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите файл", './', 'Images (*.jpg *.png *.jpeg)')
        self.MaskReadyPath.setText(str(MaskDirectory[0]))
        self.main_pixmap = QPixmap(str(MaskDirectory[0]))
        self.main_pixmap = self.main_pixmap.scaledToHeight(self.PhotoShow.height())
        self.PhotoShow.setPixmap(self.main_pixmap)

    def next(self):
        self.stackedWidget.setCurrentIndex(3)
        self.showMaximized()

    def back2(self):
        self.stackedWidget.setCurrentIndex(0)

    def back3(self):
        self.stackedWidget.setCurrentIndex(2)

    def simple_regime(self):
        self.stackedWidget.setCurrentIndex(1)

    def watermark_regime(self):
        self.stackedWidget.setCurrentIndex(2)

    def exit_app(self):
        self.close()

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.VideoPlay.setIcon(QIcon('pause.png'))
        else:
            self.VideoPlay.setIcon(QIcon('play.png'))

    def positionChanged(self, position):
        self.horizontalSlider.setValue(position)

    def durationChanged(self, duration):
        self.horizontalSlider.setRange(0, duration)

    def handleError(self):
        self.VideoPlay.setEnabled(False)
        QtWidgets.QMessageBox.warning(None, "Внимание", "Ошибка: " + self.mediaPlayer.errorString())

    def save_choose(self):
        self.SavePath.clear()
        SaveDirectory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку")
        if SaveDirectory:
            self.SavePath.setText(str(SaveDirectory))
            self.SavePath.adjustSize()

    def save_choose2(self):
        self.SavePath2.clear()
        SaveDirectory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку")
        if SaveDirectory:
            self.SavePath2.setText(str(SaveDirectory))
            self.SavePath2.adjustSize()

    def video_choose(self):
        self.VideoPath.clear()
        VideoDirectory = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите файл", './', 'Files (*.mp4 *.avi *.mov *.mkv *.mpg)')
        if VideoDirectory:
            self.VideoPath.setText(str(VideoDirectory[0]))
            self.VideoPath.adjustSize()

    def mask_choose(self):
        self.MaskPath.clear()
        MaskDirectory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку")
        if MaskDirectory:
            self.MaskPath.setText(str(MaskDirectory))
            self.MaskPath.adjustSize()

    def start_process(self):
        if self.Low_quality.isChecked():
            self.stackedWidget.setCurrentIndex(5)
            self.calc = External("e2fgvi", self.VideoPath.toPlainText(), "release_model/E2FGVI-CVPR22.pth", self.MaskPath.toPlainText(), self.SavePath.toPlainText())
            self.calc.countChanged.connect(self.onCountChanged)
            self.calc.maxValPass.connect(self.onMaxVal)
            self.calc.endSignal.connect(self.endProcess)
            self.calc.start()
        elif self.Any_quality.isChecked():
            self.stackedWidget.setCurrentIndex(5)
            self.calc = External("e2fgvi_hq", self.VideoPath.toPlainText(), "release_model/E2FGVI-HQ-CVPR22.pth", self.MaskPath.toPlainText(), self.SavePath.toPlainText())
            self.calc.countChanged.connect(self.onCountChanged)
            self.calc.maxValPass.connect(self.onMaxVal)
            self.calc.endSignal.connect(self.endProcess)
            self.calc.start()
        else:
            print("debug")
            QtWidgets.QMessageBox.warning(None, "Внимание", "Вы не выбрали разрешение")

    def video_choose2(self):
        VideoDirectory = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите файл", './', 'Files (*.mp4 *.avi *.mov *.mkv *.mpg)')
        if VideoDirectory:
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(str(VideoDirectory[0]))))
            self.VideoPlay.setEnabled(True)
            self.VideoName.setText(str(VideoDirectory[0]))
            self.play()
            self.premask_frame = get_frame_from_video(str(VideoDirectory[0]), True)

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def onCountChanged(self, value):
        self.progressBar.setValue(value)

    def onMaxVal(self, value):
        self.progressBar.setMaximum(value)

    def endProcess(self, val):
        self.stackedWidget.setCurrentIndex(val)
        QtWidgets.QMessageBox.about(None, "Успех", "Видео сохранено")


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ExampleApp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()