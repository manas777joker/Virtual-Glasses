


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

        self.glass_color = None
        self.frame_color = None
        self.side_color = None

        self.prev_glass_color = None
        self.prev_frame_color = None
        self.prev_side_color = None

        self.root = tk.Tk()
        self.l_eye = self.r_eye = None
        self.glass_line_pt1 = self.glass_line_pt2 = None
        self.cam = video.VideoStream(src=0).start()

    def start_camera_app(self):
        '''
            Start camera app by building gui interface and starting tkinter eventloop
        '''
        self._create_interface()
        self.root.after(0, func=lambda: self.update_all())
        self.root.mainloop()

    def _from_rgb(self, rgb):
        '''
            translates an rgb tuple to a int tkinter friendly color code
        '''
        return "#%02x%02x%02x" % tuple(map(int, rgb))[::-1]

    def _create_interface(self):
        '''
            Creating image display window and buttons on
        '''
        self.root.title("Try it Before you WEAR it ...")
        self.image_label = tk.Label(master=self.root, text="OpenCV")  # label for the video frame
        self.image_label.pack()
        tk.Label(self.root, text="CHOOSE COLOR FROM BELOW", fg="light green",
                 bg="#495647", font="Helvetica 16 bold italic").pack(fill="both")

        self.quit_button = tk.Button(master=self.root, text='Quit',
                                     command=self._quit, bg='#BA5843',  width=10, height=2)
        self.quit_button.pack(fill="both", side=tk.RIGHT)
        self.change_glass_btn = tk.Button(master=self.root, text='CHANGE GLASS',
                                          command=self._change_glass_color, bg='#55BA43')
        self.change_glass_btn.pack(fill="both", side=tk.LEFT)

        self.change_frame_btn = tk.Button(master=self.root, text='CHANGE FRAME',
                                          command=self._change_frame_color, bg='#D2691E')
        self.change_frame_btn.pack(fill="both", side=tk.LEFT)

        self.change_sideframe_btn = tk.Button(master=self.root, text='CHANGE SIDEVIEW',
                                              command=self._change_side_color, bg='#7B68EE')
        self.change_sideframe_btn.pack(fill="both", side=tk.LEFT)

        self.clear_effect = tk.Button(master=self.root, text='CLEAR',
                                      command=self._clear_effect, bg='#BDB76B', width=10)
        self.clear_effect.pack(fill="both", side=tk.RIGHT)

    def _clear_effect(self):
        '''
            Removing the effect
        '''
        self.glass_color = self.prev_glass_color = None
        self.change_glass_btn.config(bg='#55BA43')
        self.change_frame_btn.config(bg='#D2691E')
        self.change_sideframe_btn.config(bg='#7B68EE')
        self.root.update()

    def _change_side_color(self):
        '''
            Changing the side color of frame
        '''
        self.prev_side_color = self.side_color
        side_color = askcolor()
        self.side_color = side_color[0][::-1] if side_color[0] else self.prev_side_color
        if self.side_color:
            self.change_sideframe_btn.config(bg=self._from_rgb(self.side_color))
        self.root.update()

    def _change_glass_color(self):
        '''
            Changing the color of glasses
        '''
        self.prev_glass_color = self.glass_color
        glass_color = askcolor()
        self.glass_color = glass_color[0][::-1] if glass_color[0] else self.prev_glass_color
        if self.glass_color:
            self.change_glass_btn.config(bg=self._from_rgb(self.glass_color))
        self.root.update()

    def _change_frame_color(self):
        '''
            Changing the rounded frame color
        '''
        self.prev_frame_color = self.frame_color
        frame_color = askcolor()
        self.frame_color = frame_color[0][::-1] if frame_color[0] else self.prev_frame_color
        if self.frame_color:
            self.change_frame_btn.config(bg=self._from_rgb(self.frame_color))
        self.root.update()

    def _find_offset(self, shape):
        '''
            This is where magic happens. We are finding offsets to add in eye landmarks found by 68 points landmarks model.
            We are using this offset to create 10 points from eye's six landmarks for each eye
        '''
        return ((shape[39] - shape[36]) / 2).astype('int')[0], ((shape[45] - shape[42]) / 2).astype('int')[0]

    def fill_glasses(self, image, shape):
        '''
            Drawing glass ellipse fitting to all calculated pts and filling it.
        '''
        left_offset, right_offset = self._find_offset(shape)
        EyeEllipse = EyePoints(shape, left_offset, right_offset)
        self.left_ellipse, self.right_ellipse = EyeEllipse.find_eye_points()
        self.l_eye, self.r_eye = cv2.fitEllipse(
            self.left_ellipse), cv2.fitEllipse(self.right_ellipse)
        self.glass_line_pt1, self.glass_line_pt2 = EyeEllipse._frame_line()
        self.glass_leftside_line, self.glass_rightside_line = EyeEllipse._frame_side_line()
        cv2.ellipse(image, self.l_eye, self.glass_color, -1)
        cv2.ellipse(image, self.r_eye, self.glass_color, -1)
        return image

    def draw_frame(self, image):
        '''
            Drawing glass frame
        '''
        if self.frame_color:
            self.left_ellipse_half, self.right_ellipse_half = self.left_ellipse[
                :6], self.right_ellipse[:6]
            self.left_ellipse_half, self.right_ellipse_half = self.left_ellipse_half.astype(
                np.int32), self.right_ellipse_half.astype(np.int32)
            self.left_ellipse_half, self.right_ellipse_half = self.left_ellipse_half.reshape(
                (-1, 1, 2)), self.right_ellipse_half.reshape((-1, 1, 2))

            cv2.polylines(image, [self.left_ellipse_half, self.right_ellipse_half],
                          False, self.frame_color, 2, cv2.LINE_AA)
            cv2.polylines(image, [self.left_ellipse_half[1:5], self.right_ellipse_half[1:5]],
                          False, self.frame_color, 4, cv2.LINE_AA)
        return image

    def draw_side_frame(self, image_copy):
        '''
            Side frame for glasses
        '''
        if self.side_color:
            # side lines
            cv2.line(image_copy, self.glass_leftside_line[0],
                     self.glass_leftside_line[1], self.side_color, 3)
            cv2.line(image_copy, self.glass_rightside_line[0],
                     self.glass_rightside_line[1], self.side_color, 3)
        return image_copy

    def face_detection(self, image):
        '''
            Face detection and facial landmarks points
        '''
        image_copy = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_hog = self.face_detector(gray, 0)
        for (i, face) in enumerate(faces_hog):
            # Finding points for rectangle to draw on face
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            # Make the prediction and transfom it to numpy array
            shape = self.predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            image = self.fill_glasses(image, shape)
            image = self.draw_frame(image)
            image_copy = self.draw_side_frame(image_copy)

            alpha = 0.5
            cv2.addWeighted(image, alpha, image_copy, 1 - alpha, 0, image_copy)

            # middle line for glasses
            cv2.line(image_copy, self.glass_line_pt1,
                     self.glass_line_pt2, self.glass_color[::-1], 2)

        return image_copy

    def update_image(self):
        f = self.cam.read()
        if not self.glass_color:
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


class EyePoints():
    '''
        This is the main class which will calculate the points to make a eye glasses.
        1. First find eye facial landmarks (from 36 to 48 for both eyes)
        2. Calculate offset
        3. Calculate 10 drawing pts for each eye for fitting eyes glasses (So we will get 20 pts for both eyes by below calculation)
        4. Calculate 2 drawing pts for middle line joining the glasses.
        5. Calculate 4 drawing pts for side line joining the glasses to face (2 for each eye).
    '''

    def __init__(self, shape, left_offset, right_offset):
        self.shape = shape
        self.left_offset = left_offset
        self.right_offset = right_offset
        self.left_eye_raw_pts = self.shape[36:42]
        self.right_eye_raw_pts = self.shape[42:48]
        self.left_eye_pts = []
        self.right_eye_pts = []

    def _left_eye(self):
        pts1 = self.left_eye_raw_pts[0] - (self.left_offset, 0)
        pts2 = self.left_eye_raw_pts[0] - (0, self.left_offset)

        pts3 = self.left_eye_raw_pts[1] - (0, self.left_offset)
        pts4 = self.left_eye_raw_pts[2] - (0, self.left_offset)

        pts5 = self.left_eye_raw_pts[3] - (0, self.left_offset)
        pts6 = self.left_eye_raw_pts[3] + (self.left_offset, 0)
        pts7 = self.left_eye_raw_pts[3] + (0, self.left_offset)

        pts8 = self.left_eye_raw_pts[4] + (0, int(self.left_offset * 1.8))
        pts9 = self.left_eye_raw_pts[5] + (0, int(self.left_offset * 1.8))

        pts10 = self.left_eye_raw_pts[0] + (0, self.left_offset)
        self.left_eye_pts.extend([pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9, pts10])

    def _right_eye(self):
        pts1 = self.right_eye_raw_pts[0] - (self.right_offset, 0)
        pts2 = self.right_eye_raw_pts[0] - (0, self.right_offset)

        pts3 = self.right_eye_raw_pts[1] - (0, self.right_offset)
        pts4 = self.right_eye_raw_pts[2] - (0, self.right_offset)

        pts5 = self.right_eye_raw_pts[3] - (0, self.right_offset)
        pts6 = self.right_eye_raw_pts[3] + (self.right_offset, 0)
        pts7 = self.right_eye_raw_pts[3] + (0, self.right_offset)

        pts8 = self.right_eye_raw_pts[4] + (0, int(self.right_offset * 1.8))
        pts9 = self.right_eye_raw_pts[5] + (0, int(self.right_offset * 1.8))

        pts10 = self.right_eye_raw_pts[0] + (0, self.right_offset)
        self.right_eye_pts.extend([pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9, pts10])

    def _frame_line(self):
        '''
            Calculation of pts for frame middle line
        '''
        pt1 = self.left_eye_raw_pts[3] + (self.left_offset, 0)
        pt2 = self.right_eye_raw_pts[0] - (self.left_offset, 0)
        return tuple(pt1), tuple(pt2)

    def _frame_side_line(self):
        '''
            Calculation of pts for frame side lines
        '''
        pt1 = self.left_eye_raw_pts[0] - (self.left_offset, 0)
        pt2 = self.shape[0]
        left_side_pts = (tuple(pt1), tuple(pt2))

        pt3 = self.right_eye_raw_pts[3] + (self.right_offset, 0)
        pt4 = self.shape[16]
        right_side_pts = (tuple(pt3), tuple(pt4))
        return left_side_pts, right_side_pts

    def find_eye_points(self):
        '''
            Returning calculated eye points
        '''
        self._left_eye()
        self._right_eye()
        return np.array(self.left_eye_pts), np.array(self.right_eye_pts)


CameraApp = MyCameraApp()
CameraApp.start_camera_app()
