"""CV frame read and display"""
import sys
import cv2

class InputStream():
    def __init__(self, video_file):
        self.video_file = video_file
        self.cap = cv2.VideoCapture(self.video_file)
        self.frame_count = 0 

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        while True:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.frame_count += 1
                    return frame, self.frame_count
                else:
                    break

class DisplayStream():
    def __init__(self):
        pass

    def __del__(self):
        cv2.destroyAllWindows()

    def add_frame(self, frame, window_name):
        """Press Q to close window anytime"""
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit()

        #resize to 720p so it doesn't take up entire screen
        frame_out = cv2.resize(frame, (1280, 720))
        cv2.imshow(window_name, frame_out)
