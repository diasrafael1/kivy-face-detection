from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.lang import Builder
from jnius import autoclass
import numpy as np
import cv2

CameraInfo = autoclass('android.hardware.Camera$CameraInfo')
CAMERA_INDEX = {'front': CameraInfo.CAMERA_FACING_FRONT, 'back': CameraInfo.CAMERA_FACING_BACK}
Builder.load_file("myapplayout.kv")

class AndroidCamera(Camera):
    resolution = (640, 480)
    index = CAMERA_INDEX['back']
    counter = 0

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # age_net = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")
    # gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")
    # age_list = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
    # gender_list = ['Male', 'Female']

    def on_tex(self, *l):
        if self._camera._buffer is None:
            return None

        super(AndroidCamera, self).on_tex(*l)
        self.texture = Texture.create(size=np.flip(self.resolution), colorfmt='rgb')
        frame = self.frame_from_buf()
        # frame = self.detect_age_and_gender(frame)  # Chama a função para detecção de idade e gênero
        self.frame_to_screen(frame)

    def frame_from_buf(self):
        w, h = self.resolution
        frame = np.frombuffer(self._camera._buffer.tostring(), 'uint8').reshape((h + h // 2, w))
        frame_bgr = cv2.cvtColor(frame, 93)
        if self.index:
            return np.flip(np.rot90(frame_bgr, 1), 1)
        else:
            return np.rot90(frame_bgr, 3)

    # def detect_age_and_gender(self, frame):
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #     font = cv2.FONT_HERSHEY_SIMPLEX

    #     for (x, y, w, h) in faces:
    #         face_img = frame[y:y+h, x:x+w].copy()
    #         blob2 = cv2.dnn.blobFromImage(face_img, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    #         # Predict gender
    #         self.gender_net.setInput(blob2)
    #         gender_preds = self.gender_net.forward()
    #         gender = self.gender_list[gender_preds[0].argmax()]

    #         # Predict age
    #         self.age_net.setInput(blob2)
    #         age_preds = self.age_net.forward()
    #         age = self.age_list[age_preds[0].argmax()]

    #         overlay_text = "%s, %s" % (gender, age)
    #         cv2.putText(frame, overlay_text ,(x, y), font, 1,(255, 0, 0), 2, cv2.LINE_AA)
    #         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return frame

    def frame_to_screen(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(frame_rgb, str(self.counter), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        self.counter += 1
        flipped = np.flip(frame_rgb, 0)
        buf = flipped.tobytes()
        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

class MyLayout(BoxLayout):
    pass

class MyApp(App):
    def build(self):
        return MyLayout()

if __name__ == '__main__':
    MyApp().run()