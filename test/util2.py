import numpy as np
import cv2

refPt = []
clickPt = []
counter = 0
cropping = False
last_point = (0,0)
first = True

def show_crop(image):
	global clickPt

	def click_and_crop(event, x, y, flags, param):
		global refPt, clickPt, cropping, counter, last_point, first
		if event == cv2.EVENT_LBUTTONDOWN:
			last_point = (x, y)
			clickPt.append((x, y))
			refPt.append((x, y))
			# refPt = [(x, y)]
			cropping = True
		elif event == cv2.EVENT_LBUTTONUP:
			refPt.append((x, y))
			cropping = False
			if first:
				cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
				del clickPt[-1]
				first = False
			else:
				cv2.circle(image, clickPt[counter], 2, (0, 255, 0))
				counter += 1
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
			return None, None

		elif key == ord("f"):
			cv2.destroyAllWindows()
			return "finished", "fini"

	# close all open windows
	cv2.destroyAllWindows()
	clickPt = map(lambda x: cv2.KeyPoint(x[0], x[1], 10), clickPt)
	return refPt, clickPt
