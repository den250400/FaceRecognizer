from IPython.display import clear_output
import cv2
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video_source', type=str, default="0", help='Path to video or "0" if you want to use webcam')
args = parser.parse_args()

if args.video_source == "0":
    cap = cv2.VideoCapture(int(args.video_source))
else:
    cap = cv2.VideoCapture(args.video_source)
    
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (640, 480))
        system.predictFaces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()