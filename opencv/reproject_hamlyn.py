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

cam_to_world = np.linalg.inv(R1)


def read_disp_map(filename):
    file = open(filename)
    w,h = [int(value) for value in file.readline().split()]

    data = np.zeros(shape=(h,w),dtype=np.float32)

    for y in range(h):
        for x in range(w):
            data[y][x] = float(file.readline())

    return data


for i in range(0,20):
    disp_filename = "../hamlyn/out/{}.disp".format(i)
    disp_map = read_disp_map(disp_filename)

    #points_filename = "../hamlyn/results/result.xyz".format(PACKAGE,PACKAGE,i)

    points = cv2.reprojectImageTo3D(
        disparity=disp_map,
        Q=Q,
    )
    unrectified_points = np.zeros(points.shape, dtype=points.dtype)


    for y in range(points.shape[0]):
        for x in range(points.shape[1]):
            ux, uy = rectification_maps[0][0][y][x]
            if(ux < 0 or ux > points.shape[1]):
                continue
            if(uy < 0 or uy > points.shape[0]):
                continue
            unrectified_points[int(uy)][int(ux)] = points[y][x]

    points_file = open(points_filename,"w")

    for y in range(points.shape[0]):
        for x in range(points.shape[1]):
            #point = points[y][x]
            point = unrectified_points[y][x]
            point = np.matmul(cam_to_world,point)
            if(np.isnan(point[2] or np.isinf(point[2]))):
                continue
            if point[2] < 1 or point[2] > 1000:
                continue
            points_file.write("{} {} {} {} {}\n".format(x,y,point[0],point[1],point[2]))