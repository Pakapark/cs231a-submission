import numpy as np
import cv2

refPt = []
cropping = False

def show_crop(image):

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

    clone = image.copy()
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
    		break

        # if the 'n' key is pressed, return None
        elif key == ord("n"):
            cv2.destroyAllWindows()
            return None

        elif key == ord("f"):
            cv2.destroyAllWindows()
            return "finished"

    # close all open windows
    cv2.destroyAllWindows()
    return refPt
