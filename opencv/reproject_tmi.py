import cv2
import numpy as np


BASE_PATH = "/org/share/MediData/MedData/TMI Dataset/"
PACKAGE = "Stereo_SD_d_all"
#PACKAGE = "Stereo_SD_d_distance"
OUT_PATH = "tmi/out"

def read_disp_map(filename):
    file = open(filename)
    w,h = [int(value) for value in file.readline().split()]

    data = np.zeros(shape=(h,w),dtype=np.float32)

    for y in range(h):
        for x in range(w):
            data[y][x] = float(file.readline())

    return data


for i in range(1,36):
    calibration_filename = "{}{}/{}_{}/{}_{}_Calibration.txt".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i)
    calibration_file = open(calibration_filename)

    disp_filename = "../tmi/out/{}.disp".format(i)
    disp_map = read_disp_map(disp_filename)

    points_filename = "../tmi/results/{}/{}_{}_result.xyz".format(PACKAGE,PACKAGE,i)

    for _ in range(3):
        calibration_file.readline()

    camera_matrices = []
    distortions = []
    rotations = []
    translations = []

    for cam_i in range(2):
        values = calibration_file.readline().split()
        values = [float(value) for value in values]

        w,h, fx, _, cx, _, fy, cy, _, _, _, \
        k1, k2, d1, d2, k3, k4, k5, k6, \
        r00, r01, r02, r10, r11, r12, r20, r21,r22, t0, t1, t2 \
                = values

        w = int(w)
        h = int(h)

        calibration_file.readline()

        camera_matrices.append ( np.array((
            (fx,0,cx),
            (0,fy,cy),
            (0,0,1)), dtype=np.float)
        )

        distortions.append( np.array((k1,k2,d1,d2,k3,k4,k5,k6), dtype=np.float) )

        rotations.append( np.array((
            (r00, r01, r02),
            (r10, r11, r12),
            (r20, r21, r22)
        ), dtype=np.float))

        translations.append( np.array((t0,t1,t2),dtype=np.float) )

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

    points = cv2.reprojectImageTo3D(
        disparity=disp_map,
        Q=Q,
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

    unrectified_points = np.zeros(points.shape, dtype=points.dtype)


    cam_to_world = np.linalg.inv(R1)

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

