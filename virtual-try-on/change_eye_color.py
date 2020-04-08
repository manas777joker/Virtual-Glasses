

import numpy as np
import cv2
from PIL import Image, ImageTk
import time
import tkinter as tk
from tkinter.colorchooser import *
import dlib
from imutils import face_utils, video


class MyCameraApp():

    def __init__(self):
        # initialize hog + svm based face detector
        self.face_detector = dlib.get_frontal_face_detector()
        # landmark predictor
        self.predictor = dlib.shape_predictor(
            "./shape_predictor_68_face_landmarks.dat")

        self.color = None
        self.is_color = False
        self.root = tk.Tk()

    def start_camera_app(self):
        self._create_interface()
        self.root.after(0, func=lambda: self.update_all())
        self.root.mainloop()

    def _create_interface(self):
        self.cam = video.VideoStream(src=0).start()
        self.root.title("Change Your Eye Color using Deep Learning ...")
        self.image_label = tk.Label(master=self.root, text="OpenCV")  # label for the video frame
        self.image_label.pack()
        tk.Label(self.root, text="CHOOSE COLOR", fg="light green",
                 bg="#495647", font="Helvetica 16 bold italic").pack(fill="both")

        # quit button
        self.quit_button = tk.Button(master=self.root, text='Quit',
                                     command=self._quit, bg='#BA5843',  width=10)
        self.quit_button.pack(fill="both", side=tk.RIGHT)
        self.change_button = tk.Button(master=self.root, text='CHANGE COLOR',
                                       command=self._change_color, bg='#55BA43')
        self.change_button.pack(fill="both", side=tk.LEFT)

        # self.brightness_scale = tk.Scale(
        # self.root, from_=1, to=10, orient=tk.HORIZONTAL)
        # self.brightness_scale.pack()

    def _change_color(self):
        self.color = askcolor()
        self.color = self.color[0][::-1] if self.color else (0, 0, 200)
        self.is_color = True
        self.root.update()

    def face_detection(self, image):
        image_copy = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_hog = self.face_detector(gray, 0)
        for (i, face) in enumerate(faces_hog):
            # Finding points for rectangle to draw on face
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            # Make the prediction and transfom it to numpy array
            shape = self.predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            left_eye, right_eye = shape[36:42], shape[42:48]
            left_eye_pts, right_eye_pts = np.array(
                left_eye, np.int32), np.array(right_eye, np.int32)
            left_eye_pts, right_eye_pts = left_eye_pts.reshape(
                (-1, 1, 2)), right_eye_pts.reshape((-1, 1, 2))
            cv2.fillPoly(image, [left_eye_pts, right_eye_pts], self.color)
            alpha = 0.25
            cv2.addWeighted(image, alpha, image_copy, 1 - alpha, 0, image_copy)
        return image_copy

    def update_image(self):
        f = self.cam.read()
        if not self.is_color:
            rgb_img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            a = Image.fromarray(rgb_img)
            b = ImageTk.PhotoImage(image=a)
            self.image_label.configure(image=b)
            self.image_label._image_cache = b  # avoid garbage collection
        else:
            image = self.face_detection(f)
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            a = Image.fromarray(rgb_img)
            b = ImageTk.PhotoImage(image=a)
            self.image_label.configure(image=b)
            self.image_label._image_cache = b  # avoid garbage collection
        self.root.update()

    def update_all(self):
        self.update_image()
        self.root.after(10, func=lambda: self.update_all())

    def _quit(self):
        self.root.destroy()


CameraApp = MyCameraApp()
CameraApp.start_camera_app()
