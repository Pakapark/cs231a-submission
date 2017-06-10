from __future__ import division
from interface import show_crop
from math import ceil, sqrt
import numpy as np
import cv2, sys, parse
import random

def printIndexInfo():
    print "Please indicate the index."
    print "Index = 0 => run-across-screen-v0.1.mp4"
    print "Index = 1 => run-out-of-screen-v0.1.mp4"
    print "Index = 2 => walk-across-screen-v0.1.mp4"
    print "Index = 3 => walk-in-screen-short-v0.1.mp4"
    print "Index = 4 => walk-in-screen-v0.1.mp4"
    print "Index = 5 => walk-out-of-screen-v0.1.mp4"

def crop_image(data, frame_forward=5):
    cv2.namedWindow("crop")
    crop_map = dict()
    for frame in xrange(0, data.shape[0], frame_forward):
        data_frame = data[frame]
        refPt = show_crop(data_frame)
        if refPt is None: continue
        if refPt == "finished": break
        crop_map[frame] = refPt
    return crop_map

def bLabel(randX, randY, minX, minY, maxX, maxY, sieve=7):
    result = 0
    for i in xrange(randX, randX+sieve):
        if i > maxX or i < minX:
            result -= sieve
            continue
        for j in xrange(randY, randY+sieve):
            if j > minY and j < maxY:
                result += 1
            else:
                result -= 1
    return result/(sieve**2)

def simulateSample(frame_data, refPt , extension=15, sieve=7):
    print refPt
    num_sample = 5000
    l = 10**(-5)
    row, col, _ = frame_data.shape
    frame_data = np.mean(frame_data, axis=2)
    frame_min_x, frame_min_y = refPt[0]
    frame_max_x, frame_max_y = refPt[1]
    out_min_x, out_min_y = max(0, frame_min_x - extension), max(0, frame_min_y - extension)
    out_max_x, out_max_y = min(row, frame_max_x + extension), min(col, frame_max_y + extension)
    A = np.zeros([num_sample, sieve**2 + 2])
    b = np.zeros([num_sample, 1])
    for i in xrange(num_sample):
        randomX = random.randint(out_min_x, out_max_x - sieve)
        randomY = random.randint(out_min_y, out_max_y - sieve)
        A[i, :sieve**2] = frame_data[randomX: randomX + sieve, randomY: randomY + sieve].reshape(1, sieve**2)/256
        A[i, sieve**2] = abs((randomX + sieve/2 - (frame_min_x+frame_max_x)/2)/(frame_max_x - frame_min_x))
        A[i, sieve**2 + 1] = abs((randomY + sieve/2 - (frame_min_y+frame_max_y)/2)/(frame_max_y - frame_min_y))
        b[i] = bLabel(randomX, randomY, frame_min_x, frame_min_y, frame_max_x, frame_max_y, sieve)
        print randomX, randomY, b[i]
    # feature = np.linalg.lstsq(A, b)[0]
    feature = np.dot(np.linalg.inv(np.dot(A.T, A) + l*np.identity(A.shape[1])), np.dot(A.T, b))
        # print "residual error = " + repr(np.linalg.norm(np.dot(A, result[frame]) - b))
    return feature, (out_min_x, out_min_y, out_max_x, out_max_y)

def trackNextFrame(frame_data, feature, out_frame, sieve=7):
    frame_gray_data = np.mean(frame_data, axis=2)
    out_min_x, out_min_y, out_max_x, out_max_y = out_frame
    iter_x, iter_y = int(ceil((out_max_x - out_min_x)/sieve)), int(ceil((out_max_y - out_min_y)/sieve))
    A = np.zeros([iter_x*iter_y, sieve**2+2])
    cv2.namedWindow("NewFrame")
    for i in xrange(iter_x):
        for j in xrange(iter_y):
            temp = frame_gray_data[out_min_x + i*sieve: out_min_x + (i+1)*sieve, out_min_y + j*sieve: out_min_y + (j+1)*sieve]
            A[i*iter_y + j, :temp.shape[0]*temp.shape[1]] = temp.reshape(1, temp.shape[0]*temp.shape[1])/256
            A[i*iter_y + j, sieve**2] = abs((out_min_x + (2*i+1)/2*sieve - (out_min_x+out_max_x)/2)/(out_max_x- out_min_x))
            A[i*iter_y + j, sieve**2 + 1] = abs((out_min_y + (2*j+1)/2*sieve - (out_min_y + out_max_y)/2)/(out_max_y- out_min_y))
            # K = frame_gray_data[out_min_x + i*sieve: out_min_x + (i+1)*sieve, out_min_y + j*sieve: out_min_y + (j+1)*sieve].reshape(1, sieve**2)/256
            test = A[i*iter_y+j].dot(feature)
            if test > 1.2:
                print (int(out_min_x + (2*i-1)/2*sieve), int(out_min_y + (2*j-1)/2*sieve)), test
                cv2.circle(frame_data, (int(out_min_x + (2*i-1)/2*sieve), int(out_min_y + (2*j-1)/2*sieve)), 3, (255, 255, 0))

    while True:
        cv2.imshow("NewFrame", frame_data)
        key = cv2.waitKey(1) & 0xFF
    	if key == ord("n"):
            break
    cv2.destroyAllWindows()









def main(argv):
    # Argument Parsing
    if len(argv) < 1 or int(argv[0]) > 5 or int(argv[0]) < 0:
        printIndexInfo()
        raise Exception("Please indicate the index of file you want to observe")
    index = int(argv[0])

    filepath = parse.listFileFromDir('meta')[index]
    data = np.load(filepath)

    refPt = crop_image(np.copy(data))
    # print refPt
    use = refPt[0]
    # Assume start at frame = 0
    # for i in xrange(data.shape[0]):
    use = [(use[0][0] + 70, use[0][1]), (use[1][0] + 70, use[1][1])]
    feature, out_frame = simulateSample(data[0], use)
    trackNextFrame(data[100], feature, out_frame)
    # if i > 30:









if __name__ == "__main__":
    main(sys.argv[1:])
