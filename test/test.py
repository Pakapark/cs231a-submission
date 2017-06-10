import numpy as np
import cv2
from preprocess import parseOneVideo, listFileFromDir
from crop import crop
from mil import MIL, showAll
import sys
import kcf
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression

PROCESSED_VIDEO_DIRECTORY = 'video-data/processed'
META_DIRECTORY = 'meta'

def printFiles():
    PROCESSED_VIDEO_DIRECTORY = 'video-data/processed'
    processedFiles = listFileFromDir(PROCESSED_VIDEO_DIRECTORY)
    print "Available files are"
    for f in processedFiles:
        filename = f.split('/')[-1].replace('.mp4', '')
        if filename == ".DS_Store": continue
        print "- " + filename
    raise Exception("Please indicate the index of file you want to observe")

def human_detect(hog, image):
    rects, weights = hog.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    return pick

def main(argv):
    if len(argv) < 1: printFiles()
    filename = argv[0]
    video_filepath, data = parseOneVideo(filename)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    labels = crop(data[0])
    MILlist = [MIL(video_filepath,0,label) for label in labels]
    KCFlist = [kcf.KCF(video_filepath,0,label) for label in labels]
    # showAll(MILlist)
    kcf.showAll(KCFlist+MILlist)

if __name__ == "__main__":
    main(sys.argv[1:])
