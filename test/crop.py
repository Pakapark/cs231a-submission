import cv2
import numpy as np

refPt = []
cropping = False

def crop(image):
    result = []
    def click_and_crop(event, x, y, flags, param):
        global refPt, cropping
    	if event == cv2.EVENT_LBUTTONDOWN:
    		refPt = [(x, y)]
    		cropping = True
    	elif event == cv2.EVENT_LBUTTONUP:
    		refPt.append((x, y))
    		cropping = False
    		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
    		cv2.imshow("image", image)
    clone = np.copy(image)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    while True:
    	cv2.imshow("image", image)
    	key = cv2.waitKey(1) & 0xFF

    	# if the 'r' key is pressed, reset the cropping region
    	if key == ord("r"):
            image = clone.copy()

    	# if the 's' key is pressed, break from the loop
        elif key == ord("s"):
            result.append((min(refPt[0][0], refPt[1][0]), min(refPt[0][1], refPt[1][1]), abs(refPt[1][0] - refPt[0][0]), abs(refPt[1][1] - refPt[0][1])))

        # if the 'n' key is pressed, return None
        elif key == ord("n"):
            cv2.destroyAllWindows()
            return result
