import numpy as np
import cv2
import matplotlib.pyplot as plt

rot_angle = 28
 
template = cv2.imread('coin.jpg',cv2.IMREAD_GRAYSCALE)          # queryImage
probe = cv2.imread('coin.jpg',cv2.IMREAD_GRAYSCALE) # trainImage

# dividing height and width by 2 to get the center of the image
height, width = probe.shape[:2]
# get the center coordinates of the image to create the 2D rotation matrix
center = (width/2, height/2)
 
# using cv2.getRotationMatrix2D() to get the rotation matrix
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rot_angle, scale=1)
 
# rotate the image using cv2.warpAffine
probe = cv2.warpAffine(src=probe, M=rotate_matrix, dsize=(width, height))

w, h = template.shape[::-1]
 
# Initiate ORB detector
orb = cv2.ORB_create()
 
# find the keypoints and descriptors with ORB
probe_kp, probe_des = orb.detectAndCompute(probe, None)
template_kp, template_des = orb.detectAndCompute(template, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(template_des, probe_des)
matches = sorted(matches, key=lambda x: x.distance)


template_pts = np.float32([template_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
img_pts = np.float32([probe_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)


M, _ = cv2.findHomography(template_pts, img_pts, cv2.RANSAC)

corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
transformed_corners = cv2.perspectiveTransform(corners, M)

rect = cv2.minAreaRect(transformed_corners)
box = cv2.boxPoints(rect).astype(int)

angle = 90 - rect[2]
 
# Draw first 10 matches.
img3 = cv2.drawMatches(template,template_kp,probe,probe_kp,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.drawContours(img3, [box], 0, 255, 2)
cv2.putText(img3, f"Detected Angle: {angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 2)
cv2.putText(img3, f"Rotated Angle: {rot_angle:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 2)

plt.imshow(img3),plt.show()
