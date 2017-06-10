import numpy as np
import os
import cv2

PROCESSED_VIDEO_DIRECTORY = 'video-data/processed'
META_DIRECTORY = 'meta'
RUN_OUT_OF_SCREEN_FILE = 'run-out-of-screen-v0.1.mp4'
WALK_IN_SCREEN_SHORT = 'walk-in-screen-short-v0.1.md4'

# List all filepaths from PROCESSED_VIDEO_DIRECTORY
def listFileFromDir(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)]

# Save each video_filepath and save it as save_filename in .npy format
# in META_DIRECTORY
def saveVideoData(video_filepath, save_filename):
    cap = cv2.VideoCapture(video_filepath)
    data = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret: break
        data.append(frame)
    cap.release()
    save_filepath = os.path.join(META_DIRECTORY, save_filename)
    np.save(save_filepath, np.array(data))

# Parse all videos in PROCESSED_VIDEO_DIRECTORY that does not exist in
# META_DIRECTORY and save them as .npy file in META_DIRECTORY.
def parseAllVideo():
    processedFile = listFileFromDir(PROCESSED_VIDEO_DIRECTORY)
    metaFile = set(map(lambda x: x.split('/')[-1].replace('.npy', ''), listFileFromDir(META_DIRECTORY)))
    for i in xrange(len(allFiles)):
        filename = allFiles[i].split('/')[-1].replace('.mp4', '')
        if filename == ".DS_Store" or filename in metaFile: continue
        saveVideoData(allFiles[i], filename)
