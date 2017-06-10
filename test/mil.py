import numpy as np
import cv2
# import parse

class MIL:
    def __init__(self, video_filepath, frameTrack, bbox):
        self.video_filepath = video_filepath
        self.currentFrame = frameTrack + 1
        self.initTrack = frameTrack
        self.bbox = bbox
        self.camera = cv2.VideoCapture(video_filepath)
        self.tracker = cv2.Tracker_create("MIL")

        for i in xrange(frameTrack + 1):
            ok, image = self.camera.read()
            if not ok:
                raise Exception("This might be the wrong file.")
        ok = self.tracker.init(image, bbox)
        if not ok:
            raise Exception("This might be the wrong file.")

    def next(self):
        ok, image = self.camera.read()
        if not ok: return None
        ok, newbox = self.tracker.update(image)
        if not ok: return None
        self.currentFrame += 1
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        return image, p1, p2, (0,256,0)

def showAll(MILlist):
    cv2.namedWindow("tracking")
    while True:
        counter = 0
        for i, mil in enumerate(MILlist):
            label = mil.next()
            if i == 0:
                image = label[0]
            if label is None:
                counter +=1
                continue
            cv2.rectangle(image, *label[1:])
        if counter == len(MILlist): break
        cv2.imshow("tracking", image)
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
    cv2.destroyAllWindows()
