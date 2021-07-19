import time
import cv2
import numpy as np
from get_display_frame import InputStream
from get_display_frame import DisplayStream
from object_detector import DNYolo
from sort_obj_tracker import track_vehicles
from collections import deque

np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
counter = []
pts = [deque(maxlen=30) for _ in range(9999)]

class VehicleTracker():

    def __init__(self, video_file):
        self.input_stream = InputStream(video_file)
        self.cv_display = DisplayStream()
        self.object_detection = DNYolo()

        self.direction_to_track = None

    def __del__(self):
        del self.input_stream
        del self.cv_display
        del self.object_detection

    def track_frame(self):
        while True:
            start_frame = time.time()

            # Get the frame to be processed from video file
            frame, frame_count= self.input_stream.get_frame()

            # Get objects detected from the frame 
            detections = self.object_detection.process_frame(frame)

            # Track vehicles only from yolo generated detections
            trackers_dict, frame_sort = track_vehicles(detections, frame)

            # Annotate track id, bounding box and label, and path
            i = int(0)
            index_id = []
            c = []

            for track in trackers_dict:
                counter.append(int(track["track_id"]))
                index_id.append(int(track["track_id"]))
                color = [int(c) for c in COLORS[index_id[i] % len(COLORS)]]
                # Bounding box
                cv2.rectangle(frame_sort, (int(track["bbox"][0]), int(track["bbox"][1])), (int(track["bbox"][2]), int(track["bbox"][3])),(color), 3)
                # Track id
                cv2.putText(frame_sort,str(int(track["track_id"])),(int(track["bbox"][0]), int(track["bbox"][1] -50)),0, 5e-3 * 150, (color),2)
                # Vehicle type
                cv2.putText(frame_sort, str(track["class"]),(int(track["bbox"][0]), int(track["bbox"][1] -20)),0, 5e-3 * 150, (color),2)

                # Center point for direction line
                i += 1
                center = (int(((track["bbox"][0])+(track["bbox"][2]))/2),int(((track["bbox"][1])+(track["bbox"][3]))/2))
                pts[int(track["track_id"])].append(center)

                thickness = 5
                cv2.circle(frame_sort,  (center), 1, color, thickness)

                # Vehicle path
                for j in range(1, len(pts[int(track["track_id"])])):
                    #skip if no points in pts queue
                    if pts[int(track["track_id"])][j - 1] is None or pts[int(track["track_id"])][j] is None:
                       continue
                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    cv2.line(frame_sort,(pts[int(track["track_id"])][j-1]), (pts[int(track["track_id"])][j]),(color),thickness)

            # Count vehicles
            count = len(set(counter))

            # FPS
            end_frame = time.time()
            fps = 1/(end_frame - start_frame)

            # Annotate counter and FPS
            cv2.putText(frame_sort, "Approx Total Southbound Vehicles Counted: {}".format(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
            cv2.putText(frame_sort, "FPS: {:.2f}".format(fps), (int(20), int(160)),0, 5e-3 * 200, (0,255,0),2)

            # Display sort output 
            self.cv_display.add_frame(frame_sort, "UI")

def main():
    # Path to video file
    video_file = "/vehicle/track/vehicle_sample_vid.mp4"

    track_vehicle = VehicleTracker(video_file)
    track_vehicle.track_frame()

if __name__ == '__main__':
    main()
