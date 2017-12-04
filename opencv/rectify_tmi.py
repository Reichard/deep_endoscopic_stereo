import cv2
import numpy as np


BASE_PATH = "/org/share/MediData/MedData/TMI Dataset/"
PACKAGE = "Stereo_SD_d_all"
#PACKAGE = "Stereo_SD_d_distance"
OUT_PATH = "tmi/out"

for i in range(1,36):
    calibration_filename = "{}{}/{}_{}/{}_{}_Calibration.txt".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i)

    left_filename = "{}{}/{}_{}/{}_{}_IMG_left.bmp".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i)
    right_filename = "{}{}/{}_{}/{}_{}_IMG_right.bmp".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i)
    left_mask_filename = "{}{}/{}_{}/{}_{}_MASK_left.png".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i)
    right_mask_filename = "{}{}/{}_{}/{}_{}_MASK_right.png".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i)
    eval_mask_filename = "{}{}/{}_{}/{}_{}_MASK_eval.png".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i)

    left_rectified_filename  = "{}{}/{}_{}/{}_{}_IMG_REC_CV_left.bmp".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i)
    right_rectified_filename = "{}{}/{}_{}/{}_{}_IMG_REC_CV_right.bmp".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i)
    left_mask_rectified_filename = "{}{}/{}_{}/{}_{}_MASK_REC_CV_left.png".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i)
    right_mask_rectified_filename = "{}{}/{}_{}/{}_{}_MASK_REC_CV_right.png".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i)
    eval_mask_rectified_filename = "{}{}/{}_{}/{}_{}_MASK_REC_eval.png".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i)


    calibration_file = open(calibration_filename)
    left_image = cv2.imread(left_filename)
    right_image = cv2.imread(right_filename)
    left_mask = cv2.imread(left_mask_filename)
    right_mask = cv2.imread(right_mask_filename)
    eval_mask = cv2.imread(eval_mask_filename)

    evmrec = cv2.imread("{}{}/{}_{}/{}_{}_MASK_REC_eval.png".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i))
    cv2.imwrite("{}{}/{}_{}/{}_{}_MASK_REC_eval_orig.png".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i), evmrec)

    #evmrec = cv2.imread("{}{}/{}_{}/{}_{}_MASK_REC_eval_orig.png".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i))
    #cv2.imwrite("{}{}/{}_{}/{}_{}_MASK_REC_eval.png".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i), evmrec)

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

    left_mask_rectified = cv2.remap(
        src=left_mask,
        map1=rectification_maps[0][0],
        map2=rectification_maps[0][1],
        interpolation=cv2.INTER_CUBIC,
    )

    right_mask_rectified = cv2.remap(
        src=right_mask,
        map1=rectification_maps[1][0],
        map2=rectification_maps[1][1],
        interpolation=cv2.INTER_CUBIC,
    )

    eval_mask_rectified = cv2.remap(
        src=eval_mask,
        map1=rectification_maps[0][0],
        map2=rectification_maps[0][1],
        interpolation=cv2.INTER_CUBIC,
    )

    #cv2.imshow("left", left_image)
    #cv2.imshow("right", right_image)
    #cv2.imshow("left rect", left_image_rectified)
    #cv2.imshow("right rect", right_image_rectified)
    #cv2.waitKey()

    cv2.imwrite(left_rectified_filename,left_image_rectified)
    cv2.imwrite(right_rectified_filename,right_image_rectified)
    cv2.imwrite(left_mask_rectified_filename,left_mask_rectified)
    cv2.imwrite(right_mask_rectified_filename,right_mask_rectified)
    cv2.imwrite(eval_mask_rectified_filename,eval_mask_rectified)

    #cv2.imshow("evalmask", eval_mask)
    #cv2.waitKey()

#cv2.reprojectImageTo3D(disparity, Q[, _3dImage[, handleMissingValues[, ddepth]]]) 3dImage