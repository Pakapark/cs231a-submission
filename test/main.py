from __future__ import division
import numpy as np
import cv2
import util, parse
import sys
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

def reject_outliers(new_kp, m=3):
    # data_x, data_y = np.array(map(lambda x: int(x[0]), new_kp)), np.array(map(lambda x: int(x[1]), new_kp))
    # index_x_sort, index_y_sort = data_x.argsort(), data_y.argsort()
    # i = 0
    # while True:
    #     index_x_sort = data_x.argsort()
    #     if abs(data_x[index_x_sort[i]] - data_x[index_x_sort[i+1]]) < 20:
    #         break
    #     np.delete(data_x, index_x_sort[0])
    #     i += 1
    #
    # j = 0
    # while True:
    #     index_y_sort = data_y.argsort()
    #     if abs(data_y[index_y_sort[j]] - data_y[index_y_sort[j+1]]) < 20:
    #         break
    #     np.delete(data_y, index_y_sort[0])
    #     j += 1
    #
    # index_x_sort, index_y_sort = data_x.argsort(), data_y.argsort()
    # return new_kp, data_x[index_x_sort[0]], data_x[index_x_sort[-1]], data_y[index_y_sort[0]], data_y[index_y_sort[-1]]

    data_x, data_y = np.array(map(lambda x: int(x[0]), new_kp)), np.array(map(lambda x: int(x[1]), new_kp))
    return np.median(data_y), np.median(data_x)
    # data_x_mean, data_y_mean = np.mean(data_x), np.mean(data_y)
    # data_x_std, data_y_std = np.std(data_x), np.std(data_y)
    # min_x, min_y = float('inf'), float('inf')
    # max_x, max_y = -float('inf'), -float('inf')
    # result = []
    # for i in xrange(data_x.shape[0]):
    #     # if data_x[i] - data_x_mean < m*data_x_std and data_y[i] - data_y_mean < m*data_y_std:
    #         # result.append((data_x[i], data_y[i]))
    #     if data_x[i] < min_x: min_x = data_x[i]
    #     if data_x[i] > max_x: max_x = data_x[i]
    #     if data_y[i] < min_y: min_y = data_y[i]
    #     if data_y[i] > max_y: max_y = data_y[i]
    # return min_x, max_x, min_y, max_y

def remove_outlier(good, kp2):
    new_kp = [kp2[mat[0].trainIdx].pt for mat in good]
    data_x, data_y = np.array(map(lambda x: int(x[0]), new_kp)), np.array(map(lambda x: int(x[1]), new_kp))
    index_x_sort, index_y_sort = data_x.argsort(), data_y.argsort()
    index_inverse_x_sort, index_inverse_y_sort = index_x_sort[::-1], index_y_sort[::-1]
    remove_index = set()

    i = 0
    while True:
        temp = len(remove_index)
        if data_x[index_x_sort[i+1]] - data_x[index_x_sort[i]] > 20:
            remove_index.add(index_x_sort[i])
        if data_y[index_y_sort[i+1]] - data_y[index_y_sort[i]] > 20:
            remove_index.add(index_y_sort[i])
        if len(remove_index) == temp: break

    j = 0
    while True:
        temp = len(remove_index)
        if data_x[index_inverse_x_sort[i+1]] - data_x[index_inverse_x_sort[i]] > 20:
            remove_index.add(index_inverse_x_sort[i])
        if data_y[index_inverse_y_sort[i+1]] - data_y[index_inverse_y_sort[i]] > 20:
            remove_index.add(index_inverse_y_sort[i])
        if len(remove_index) == temp: break

    remove_index = sorted(list(remove_index))[::-1]
    for i in remove_index:
        del good[i]

    return good

def test(data):
    cv2.namedWindow("track")
    frame = 0
    while True:
        edges = cv2.Canny(data[frame],100,200)

        while True:
            cv2.imshow("track", edges)
            k = cv2.waitKey(1) & 0xff
            if k == ord("n"): break
        frame += 1


def tracking(data, frame, refPt):
    cv2.namedWindow("track")
    sift = cv2.xfeatures2d.SIFT_create()
    minX, maxX, minY, maxY = refPt[0][1], refPt[1][1], refPt[0][0], refPt[1][0]
    width = maxX - minX
    height = maxY - minY
    print width, height

    while True:
        # sifting_image = cv2.Canny(data[frame, minX: maxX, minY: maxY],100,200)
        sifting_image = data[frame, minX: maxX, minY: maxY]
        print maxX - minX, maxY - minY
        kp, des = sift.detectAndCompute(sifting_image,None)
        # kp, des = sift.compute(sifting_image, keyPt)
        # img = cv2.drawKeypoints(sifting_image,kp,sifting_image)

        # sifting_image2 = cv2.Canny(data[frame + 1], 100, 200)
        sifting_image2 = data[frame+1]
        kp2, des2 = sift.detectAndCompute(sifting_image2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des,des2, 2)

        # Apply ratio test
        good = [[m] for m,n in matches if m.distance < 0.75 * n.distance]
        good = remove_outlier(good, kp2)
        new_kp = [kp2[mat[0].trainIdx].pt for mat in good]

        img3 = cv2.drawMatchesKnn(sifting_image,kp,sifting_image2,kp2,good,None,flags=2)
        # minY, maxY, minX, maxX = reject_outliers(new_kp)
        # medY, medX = reject_outliers(new_kp)
        # medYPrev, medXPrev = reject_outliers()

        # print minX, maxX, minY, maxY
        # cv2.rectangle(sifting_image2, (minY, minX),(maxY, maxX), (0, 256, 0))

        while True:
            cv2.imshow("track", img3)
            k = cv2.waitKey(1) & 0xff
            if k == ord("n"): break


        frame += 1


    cv2.destroyAllWindows()


def selectPedestrian(data, fast_forward=10):
    record = []
    for frame in xrange(data.shape[0]):
        if frame % fast_forward > 0: continue
        refPt = util.show_crop(data[frame])
        if type(refPt) == list: return frame, refPt
        if refPt == "finished": return record
    return record

def printIndexInfo():
    print "Please indicate the index."
    print "Index = 0 => run-across-screen-v0.1.mp4"
    print "Index = 1 => run-out-of-screen-v0.1.mp4"
    print "Index = 2 => walk-across-screen-v0.1.mp4"
    print "Index = 3 => walk-in-screen-short-v0.1.mp4"
    print "Index = 4 => walk-in-screen-v0.1.mp4"
    print "Index = 5 => walk-out-of-screen-v0.1.mp4"

def main(argv):

    # Argument Parsing
    if len(argv) < 1 or int(argv[0]) > 5 or int(argv[0]) < 0:
        printIndexInfo()
        raise Exception("Please indicate the index of file you want to observe")
    index = int(argv[0])

    # List all files in the directory and load the data as specified
    fileList = parse.listFileFromDir('meta')
    data = np.load(fileList[index])

    # Get the frame, top left corner, and top right corner from manual labelling
    frame, refPt = selectPedestrian(np.copy(data), fast_forward=10)
    # print refPt
    # print keyPt

    # Perform tracking
    tracking(data, frame, refPt)

    # test(data)



if __name__ == "__main__":
    main(sys.argv[1:])
