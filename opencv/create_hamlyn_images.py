import scipy.ndimage
from PIL import ImageOps
import matplotlib
import cv2
import numpy as np

camera_matrices = [None,None]
distortions = [None,None]
rotations = [None,None]
translations = [None,None]

w = 360
h = 288

camera_matrices[0] = np.array((
    (391.656525, 0.000000, 165.964371),
    (0.000000, 426.835144, 154.498138),
    (0.000000, 0.000000, 1.000000)
), dtype = np.float)
distortions[0] = np.array((
    -0.196312, 0.129540, 0.004356, 0.006236
), dtype = np.float)

camera_matrices[1] = np.array((
    (390.376862, 0.000000, 190.896454),
    (0.000000, 426.228882, 145.071411),
    (0.000000, 0.000000, 1.000000)
), dtype = np.float)
distortions[1] = np.array((
    -0.205824, 0.186125, 0.015374, 0.003660
), dtype = np.float)

rotations[0] = np.array((
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1)
), dtype = np.float)
translations[0] = np.array((
    0,0,0
), dtype = np.float)

rotations[1] = np.array((
    (0.999999, -0.001045, -0.000000),
    (0.001045, 0.999999, -0.000000),
    (0.000000, 0.000000, 1.000000)
), dtype = np.float)
translations[1] = np.array((
    -5.520739, -0.031516, -0.051285
), dtype = np.float)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cameraMatrix1= camera_matrices[0],
    distCoeffs1= distortions[0],
    cameraMatrix2 = camera_matrices[1],
    distCoeffs2=distortions[1],
    imageSize=(w,h),
    R=rotations[1],
    T=translations[1],
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0.7
)

rectification_maps = [[None,None],[None,None]]

rectification_maps[0][0], rectification_maps[0][1] = \
    cv2.initUndistortRectifyMap(
        cameraMatrix=camera_matrices[0],
        distCoeffs=distortions[0],
        R=R1,
        newCameraMatrix=P1,
        size=(w,h),
        m1type=cv2.CV_32FC2
    )

rectification_maps[1][0], rectification_maps[1][1] = \
    cv2.initUndistortRectifyMap(
        cameraMatrix=camera_matrices[1],
        distCoeffs=distortions[1],
        R=R2,
        newCameraMatrix=P2,
        size=(w,h),
        m1type=cv2.CV_32FC2
    )


left_video_filename = "/org/share/MediData/MedData/Hamlyn Dataset/f5_dynamic_deint_L.mp4"
right_video_filename = "/org/share/MediData/MedData/Hamlyn Dataset/f5_dynamic_deint_L.mp4"

frames_path = "/org/share/MediData/MedData/Hamlyn Dataset/"

left_video = cv2.VideoCapture(left_video_filename)
right_video = cv2.VideoCapture(right_video_filename)

frame = 0
success_left,left_image = left_video.read()
success_right,right_image = right_video.read()

while success_left and success_right:
    if frame >= 20:
        break

    left_image_rectified = cv2.remap(
        src=left_image,
        map1=rectification_maps[0][0],
        map2=rectification_maps[0][1],
        interpolation=cv2.INTER_CUBIC,
    )

    right_image_rectified = cv2.remap(
        src=right_image,
        map1=rectification_maps[1][0],
        map2=rectification_maps[1][1],
        interpolation=cv2.INTER_CUBIC,
    )

    cv2.imwrite("{}/left_{}.png".format(frames_path, frame), left_image )
    cv2.imwrite("{}/right_{}.png".format(frames_path, frame), right_image )
    cv2.imwrite("{}/left_rec_{}.png".format(frames_path, frame), left_image_rectified )
    cv2.imwrite("{}/right_rec_{}.png".format(frames_path, frame), right_image_rectified )

    success_left,left_image = left_video.read()
    success_right,right_image = right_video.read()
    frame += 1