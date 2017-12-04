import cv2, numpy as np

vidcap = cv2.VideoCapture('big_buck_bunny_720p_5mb.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print 'Read a new frame: ', success
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1


cv2.stereoRectify()

cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, map1, map2) → None¶

cv2.remap()


cv2.reprojectImageTo3D(disparity, Q[, _3dImage[, handleMissingValues[, ddepth]]]) → _3dImage