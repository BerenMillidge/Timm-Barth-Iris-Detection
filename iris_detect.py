from __future__ import division
import numpy as np
import cv2 
import itertools
import math

def cv2normalize(img):
	return cv2.normalize(img, img, -127.0, 127.0, cv2.NORM_MINMAX, cv2.CV_8S)


def sobel_gradient(img):
	sobelx = cv2.Sobel(img, cv2.CV_16S, 1,0,ksize=3)
	sobelx = cv2normalize(sobelx)
	sobely = cv2.Sobel(img, cv2.CV_16S, 0,1,ksize=3)
	sobely = cv2normalize(sobely)
	return [sobelx, sobely]

def numpy_gradient(img):
	x,y = np.gradient(img)
	x = cv2normalize(x)
	y = cv2normalize(y)
	return [x,y]

def bitwise_invert(img):
	return cv2.bitwise_not(img)

def smooth_and_invert(img, kernel=(3,3), sigma=2):
	blur = cv2.GaussianBlur(img, kernel, sigma)
	return bitwise_invert(blur)

def normalize(x,y, threshold= 0):
	magnitude = math.sqrt(x**2 + y**2)
	if magnitude > threshold:
		return [x / magnitude, y / magnitude]

	return [0,0]

def normalize_vector(vect):
	return vect / np.sqrt(np.sum(np.square(vect)))

def manual_vector_normalize(vect):
	return normalize(vect[0], vect[1])


def normalize_gradients(xgrad, ygrad, with_itertools = True):

	assert xgrad.shape == ygrad.shape, 'Gradient matrices must be same dimensions'
	assert len(xgrad.shape) == 2, 'Function only works on two dimensional grayscale images'
	h,w = xgrad.shape
	# I think this is just equivalent to:
	if with_itertools:
		gradx = []
		grady = []
		for (xrow, yrow) in itertools.izip(xgrad, ygrad):
			for (x,y) in itertools.izip(xrow, yrow):
				norm = normalize(x,y)
				gradx.append(norm[0])
				grady.append(norm[1])

		gradx = np.reshape(np.array(gradx, dtype=np.float16), (-1,w))
		grady = np.reshape(np.array(grady, dtype=np.float16), (-1,w))
		#grad = np.dstack((gradx, grady))
		return gradx, grady

	else:
		normed_grad = np.zeros((h,w))
		gradx = []
		grady = []
		for i in xrange(h):
			for j in xrange(w):
				x = xgrad[i][j]
				y = ygrad[i][j]
				gradx.append(norm[0])
				grady.append(norm[1])

		gradx = np.reshape(np.array(gradx, dtype=np.float16), (-1,w))
		grady = np.reshape(np.array(grady, dtype=np.float16), (-1,w))
		#grad = np.dstack((gradx, grady))
		return gradx, grady


def timm_iris_detect(img):
	if type(img) != type(np.zeros((1,1))):
		img = np.array(img) 
	assert len(img.shape) == 2, 'Code currently only applies to two dimensional grayscale images'
	h,w = img.shape

	xgrad, ygrad = sobel_gradient(img)
	xgrad, ygrad = normalize_gradients(xgrad, ygrad)

	smoothed = smooth_and_invert(img)

	max_sum = -1
	center = [h//2, w//2] # default center!
	for (xcenter, ycenter) in np.ndindex(h,w):

		current_sum = 0

		for (xpos, ypos) in np.ndindex(h,w):
			print "In inner iris detect loop!"
			print h, w
			displacement = manual_vector_normalize([xpos - xcenter, ypos - ycenter])
			print "Displacement calculated!"
			dotval = displacement[0] * ygrad[xpos, ypos] + displacement[1] * xgrad[xpos, ypos]
			print "Dotval" + str(dotval)
			current_sum +=  np.abs(dotval) 
			print "CURRENT SUM!"  + str(current_sum)
			if np.isnan(current_sum):
				raise ValueError('Current sum should not diverge!')

		weight = smoothed[xcenter][ycenter]
		current_sum = weight * current_sum
		print current_sum

		if current_sum > max_sum:
			max_sum = current_sum
			center = [xcenter, ycenter]
		print "In outer iris detect loop!"

	return center