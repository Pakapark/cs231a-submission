import numpy as np
import cv2
import util, parse
import sys
import crop
from mil import *

def selectPedestrian(data, fast_forward=10):
    record = []
    for frame in xrange(data.shape[0]):
        if frame % fast_forward > 0: continue
        refPt = util.show_crop(data[frame])
        if type(refPt) == list: return (frame, (refPt[0][0], refPt[0][1], refPt[1][0] - refPt[0][0], refPt[1][1] - refPt[0][1]))
        if refPt == "finished": return record
    return record

def tracking(video_filepath, init_frame, bbox):
    cv2.namedWindow("tracking")
    camera = cv2.VideoCapture(video_filepath)
    tracker = cv2.Tracker_create("MIL")
    frame = 0

    while camera.isOpened():
        ok, image = camera.read()
        if not ok: break
        if frame == init_frame:
            ok = tracker.init(image, bbox)
        if frame >= init_frame:
            ok, newbox = tracker.update(image)
            print ok, newbox
            if ok:
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(image, p1, p2, (0,256,0))
        frame += 1

        cv2.imshow("tracking", image)
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break # esc pressed

def main(argv):
    if len(argv) < 1:
        raise Exception("Please indicate the index of file you want to observe")
    index = int(argv[0])
    fileList = parse.listFileFromDir('meta')
    data = np.load(fileList[index])
    # frameLabels = selectPedestrian(data, fast_forward=10)
    frameLabels = crop.crop(data[0])[0]
    print frameLabels
    videoFileList = parse.listFileFromDir('video-data/processed')
    video_filepath = videoFileList[index + 1]
    tracking(video_filepath, 0, frameLabels)

if __name__ == "__main__":
    main(sys.argv[1:])
