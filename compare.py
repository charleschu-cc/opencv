import numpy as np
import cv2
import sys, getopt

try:
    opts, args = getopt.getopt(sys.argv[1:], "hb:c:", ["ifile=", "ofile="])
except getopt.GetoptError:
    print 'compare.py -b <base image> -c <compare image>'
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print 'compare.py -b <base image> -c <compare image>'
        sys.exit()
    elif opt in ("-b", "--ifile"):
        basefile = arg
    elif opt in ("-c", "--ofile"):
        comparefile = arg

print 'Base image:', basefile
print 'Compare image:', comparefile

img1 = cv2.imread(basefile, 0)
img2 = cv2.imread(comparefile, 0)

#sift = cv2.AKAZE_create()
sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

print("keypoints: {}, descriptors: {}" .format(len(kp1), des1.shape))
print("keypoints: {}, descriptors: {}" .format(len(kp2), des2.shape))

# BruteForceMethod
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# FLANN parameters
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)   # or pass empty dictionary
#
# flann = cv2.FlannBasedMatcher(index_params,search_params)
#
# matches = flann.knnMatch(des1,des2,k=2)


good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None,flags=2)

print "Match points:", len(good)

cv2.imshow("Output", img3)
cv2.waitKey(0)
