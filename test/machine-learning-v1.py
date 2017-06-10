from __future__ import division
from interface import show_crop
from math import ceil, sqrt
import numpy as np
import cv2, sys, parse
import random
import misvm.misvm as misvm

classifier = misvm.MISVM(kernel='linear', C=1.0, max_iters=50)

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
    num_bag = 250
    num_sample = 30
    l = 10**(-5)
    row, col, _ = frame_data.shape
    frame_data = np.mean(frame_data, axis=2)
    frame_min_x, frame_min_y = refPt[0]
    frame_max_x, frame_max_y = refPt[1]
    out_min_x, out_min_y = max(0, frame_min_x - extension), max(0, frame_min_y - extension)
    out_max_x, out_max_y = min(row, frame_max_x + extension), min(col, frame_max_y + extension)
    A = np.zeros([num_bag*num_sample, 1, sieve**2 + 2])
    b = np.zeros([num_bag*num_sample, 1])
    c = np.zeros([num_sample, 1])
    for j in xrange(num_bag):
        for i in xrange(num_sample):
            randomX = random.randint(out_min_x, out_max_x - sieve)
            randomY = random.randint(out_min_y, out_max_y - sieve)
            A[j*num_sample + i, 0, :sieve**2] = frame_data[randomX: randomX + sieve, randomY: randomY + sieve].reshape(1, sieve**2)/256
            A[j*num_sample + i, 0, sieve**2] = abs((randomX + sieve/2 - (frame_min_x+frame_max_x)/2)/(frame_max_x - frame_min_x))
            A[j*num_sample + i, 0, sieve**2 + 1] = abs((randomY + sieve/2 - (frame_min_y+frame_max_y)/2)/(frame_max_y - frame_min_y))
            c[i] = bLabel(randomX, randomY, frame_min_x, frame_min_y, frame_max_x, frame_max_y, sieve)
            if c[i] > 0:
                b[j*num_sample + i] = -1.0
            else:
                b[j*num_sample + i] = 1.0
    # feature = np.linalg.lstsq(A, b)[0]
    # feature = np.dot(np.linalg.inv(np.dot(A.T, A) + l*np.identity(A.shape[1])), np.dot(A.T, b))
    classifier.fit(A, b)
        # print "residual error = " + repr(np.linalg.norm(np.dot(A, result[frame]) - b))
    return (out_min_x, out_min_y, out_max_x, out_max_y)

def trackNextFrame(frame_data, out_frame, sieve=7):
    frame_gray_data = np.mean(frame_data, axis=2)
    out_min_x, out_min_y, out_max_x, out_max_y = out_frame
    iter_x, iter_y = int(ceil((out_max_x - out_min_x)/sieve)), int(ceil((out_max_y - out_min_y)/sieve))
    A = np.zeros([iter_x*iter_y, 1, sieve**2+2])
    cv2.namedWindow("NewFrame")
    for i in xrange(iter_x):
        for j in xrange(iter_y):
            A[i*iter_y + j, 0, :sieve**2] = frame_gray_data[out_min_x + i*sieve: out_min_x + (i+1)*sieve, out_min_y + j*sieve: out_min_y + (j+1)*sieve].reshape(1, sieve**2)/256
            A[i*iter_y + j, 0, sieve**2] = abs((out_min_x + (2*i+1)/2*sieve - (out_min_x+out_max_x)/2)/(out_max_x- out_min_x))
            A[i*iter_y + j, 0, sieve**2 + 1] = abs((out_min_y + (2*j+1)/2*sieve - (out_min_y + out_max_y)/2)/(out_max_y- out_min_y))
            # K = frame_gray_data[out_min_x + i*sieve: out_min_x + (i+1)*sieve, out_min_y + j*sieve: out_min_y + (j+1)*sieve].reshape(1, sieve**2)/256
            # test = A[i*iter_y+j].dot(feature)
    test = classifier.predict(A)
    print test
    for i, val in enumerate(np.sign(test)):
        if val > 0:
            print (int(out_min_x + (2*i/iter_y-1)/2*sieve), int(out_min_y + (2*(i%iter_y)-1)/2*sieve)), val
            cv2.circle(frame_data, (int(out_min_x + ((2*i/iter_y-1)-1)/2*sieve), int(out_min_y + (2*(i%iter_y)-1)/2*sieve)), 3, (0, 0, 255))

    # if test > 1.2:
    #     print (int(out_min_x + (2*i-1)/2*sieve), int(out_min_y + (2*j-1)/2*sieve)), test
    #     cv2.circle(frame_data, (int(out_min_x + (2*i-1)/2*sieve), int(out_min_y + (2*j-1)/2*sieve)), 3, (0, 0, 255))

    while True:
        cv2.imshow("NewFrame", frame_data)
        key = cv2.waitKey(1) & 0xFF
    	if key == ord("n"):
            break
    cv2.destroyAllWindows()

def bagConstruction(frame_data, topLeft, bottomRight, imageWidth, imageHeight, extension=15, numSample=2000):
    print topLeft, bottomRight

    midImage = (topLeft + bottomRight)/2

    outTopLeft = topLeft - extension
    if outTopLeft[0] < 0: outTopLeft[0] = 0
    if outTopLeft[1] < 0: outTopLeft[1] = 0

    outBottomRight = topLeft + extension
    if outBottomRight[0] > frame_data.shape[0]: outTopLeft[0] = frame_data.shape[0] - 1
    if outBottomRight[1] > frame_data.shape[1]: outTopLeft[1] = frame_data.shape[1] - 1

    frame_data = np.mean(frame_data, axis=-1)
    positiveBags, negativeBags = [[]], []
    positiveMaps, negativeMaps = {}, {}
    for _ in xrange(numSample):
        randomX = random.randint(outTopLeft[0], outBottomRight[0])
        randomY = random.randint(outTopLeft[1], outBottomRight[1])
        dx = abs(randomX - midImage[0])/imageWidth
        dy = abs(randomY - midImage[1])/imageHeight
        row = list(frame_data[randomX - 1: randomX + 2, randomY - 1: randomY + 2].reshape(9)/256) + [dx, dy, dx**2, dy**2, dx*dy]
        if randomX > topLeft[0] and randomX < bottomRight[0] and randomY > topLeft[1] and randomY < bottomRight[1]:
            positiveMaps[len(positiveBags[0])] = (randomX, randomY)
            positiveBags[0].append(row)
        else:
            negativeMaps[len(negativeMaps)] = (randomX, randomY)
            negativeBags.append([row])
    labels = [1.0] + [-1.0]*len(negativeBags)
    return positiveBags, negativeBags, positiveMaps, negativeMaps, labels

def cropImage(frame_data, topLeft, bottomRight, imageWidth, imageHeight, extension=15, numSample=2000):
    print topLeft, bottomRight

    midImage = (topLeft + bottomRight)/2

    outTopLeft = topLeft - extension
    # if outTopLeft[0] < 0: outTopLeft[0] = 0
    # if outTopLeft[1] < 0: outTopLeft[1] = 0

    outBottomRight = topLeft + extension
    # if outBottomRight[0] > frame_data.shape[0]: outTopLeft[0] = frame_data.shape[0] - 1
    # if outBottomRight[1] > frame_data.shape[1]: outTopLeft[1] = frame_data.shape[1] - 1

    frame_data = np.mean(frame_data, axis=-1)
    bags = []
    bagMaps = {}
    for _ in xrange(numSample):
        randomX = random.randint(outTopLeft[0], outBottomRight[0])
        randomY = random.randint(outTopLeft[1], outBottomRight[1])
        dx = abs(randomX - midImage[0])/imageWidth
        dy = abs(randomY - midImage[1])/imageHeight
        row = list(frame_data[randomX - 1: randomX + 2, randomY - 1: randomY + 2].reshape(9)/256) + [dx, dy, dx**2, dy**2, dx*dy]
        bagMaps[len(bags)] = (randomX, randomY)
        bags.append(row)
    return bags, bagMaps

def main(argv):
    # Argument Parsing
    if len(argv) < 1 or int(argv[0]) > 5 or int(argv[0]) < 0:
        printIndexInfo()
        raise Exception("Please indicate the index of file you want to observe")
    index = int(argv[0])

    filepath = parse.listFileFromDir('meta')[index]
    data = np.load(filepath)

    refPt = crop_image(np.copy(data))
    topLeft, bottomRight = np.array(refPt[0][0]), np.array(refPt[0][1])
    imageWidth = abs(bottomRight[0] - topLeft[0])
    imageHeight = abs(bottomRight[1] - topLeft[1])
    extension = 15
    numSample = 2000
    cv2.namedWindow("NewFrame")

    # 1. Construct Positive bags and Negative bags (All Positive are in one bag and Negative bag contain only one data)
    positiveBags, negativeBags, positiveMaps, negativeMaps, labels = bagConstruction(data[0], topLeft, bottomRight, imageWidth, imageHeight, extension=extension, numSample=numSample)
    # print len(positiveBags[0]), len(negativeBags)

    # 2. Train in MISVM
    classifier.fit(positiveBags + negativeBags, labels)

    # 3. For loop on each frame
    for frame in xrange(1, data.shape[0], 3):
        # 3.1 Crop a set of image using and compute feature vector based on MIL apperance model
        bags, bagMaps = cropImage(data[frame], np.array(topLeft), np.array(bottomRight), imageWidth, imageHeight, extension=extension, numSample=numSample)
        # 3.2 Use MI Classifier to estimate p(y=1|x) => apply MILBoost
        predict = classifier.predict(bags)
        maxPoint = bagMaps[predict.argsort()[-1]]
        topLeft = tuple([int((maxPoint[0] - topLeft[0])*0.85 + topLeft[0]), topLeft[1]])
        # topLeft = tuple([int(maxPoint[0] - imageWidth/2), topLeft[1]])#int(maxPoint[1] - imageHeight/2)])
        bottomRight = tuple([int(maxPoint[0] + imageWidth/2), bottomRight[1]])#int(maxPoint[1] + imageHeight/2)])
        print "After update"
        print topLeft, bottomRight

        # 3.3 Update the location of the image
        cv2.rectangle(data[frame], topLeft, bottomRight, (0, 255, 0), 1)
        # 3.4 Crop out two images one for positive and another is negative
        positiveBags, negativeBags, positiveMaps, negativeMaps, labels = bagConstruction(data[frame], np.array(topLeft), np.array(bottomRight), imageWidth, imageHeight, extension=extension, numSample=numSample)
        # 3.5 Update MIL appearance model
        classifier.fit(positiveBags + negativeBags, labels)

        while True:
            cv2.imshow("NewFrame", data[frame])
            key = cv2.waitKey(1) & 0xFF
            if key == ord("n"):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
