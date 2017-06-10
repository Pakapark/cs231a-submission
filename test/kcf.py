import numpy as np
import cv2
import kcftracker
# import parse

class KCF:
    def __init__(self, video_filepath, frameTrack, bbox):
        self.camera = cv2.VideoCapture(video_filepath)
        self.tracker = kcftracker.KCFTracker(True, True, False)

        for i in xrange(frameTrack + 1):
            ok, image = self.camera.read()
        ok = self.tracker.init(bbox, image)

    def next(self):
        ok, image = self.camera.read()
        newbox = map(int, self.tracker.update(image))
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        return image, p1, p2, (0,256,256)

def showAll(KCFlist):
    cv2.namedWindow("tracking")
    while True:
        counter = 0
        for i, kcf in enumerate(KCFlist):
            label = kcf.next()
            if i == 0:
                image = label[0]
            if label is None:
                counter +=1
                continue
            cv2.rectangle(image, *label[1:])
        if counter == len(KCFlist): break
        cv2.imshow("tracking", image)
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
    cv2.destroyAllWindows()
