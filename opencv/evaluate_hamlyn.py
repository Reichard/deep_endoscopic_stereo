import math
import os
import numpy as np
from scipy import ndimage
from scipy import misc

BASE_PATH = "/org/share/MediData/MedData/Hamlyn Dataset/f5"
BORDER = 15

w = 360
h = 288

def read_depth(filename):
    file = open(filename)

    data = np.zeros(shape=(h,w),dtype=np.float32)

    for y in range(h):
        for x in range(w):
            data[y][x] = [float(value) for value in file.readline().split()][2]

    return data

for i in range(0,20):
    ref_filename = "{}/heartDepthMap_{}.txt".format(BASE_PATH,i)
    points_filename = "../hamlyn/results/disparityMap_{}.txt".format(i)

    if not os.path.isfile(points_filename):
        continue

    depth_map = read_depth(points_filename)
    ref_map  = read_depth(ref_filename)

    error_map = np.abs(depth_map - ref_map)
    error_map[ref_map < 0.1] = 0
    error_map[depth_map < 0.1] = 0
    error_map[depth_map > 1000] = 0
    misc.imsave("../hamlyn/results/error_{}.png".format(i), error_map)

    rmse = 0
    avg = 0
    count = 0
    max = 0
    min = 10000000

    for y in range(0,h):
        for x in range(0,w):
            depth  = depth_map[y][x]
            ref = ref_map[y][x]

            if(y < BORDER or y >= w-BORDER):
                continue

            if(depth <= 0.1 or ref <= 0.1 or depth > 100):
                continue

            error = abs(depth-ref)

            rmse += error*error
            avg += error
            count += 1

            if error < min:
                min = error
            if error > max:
                max = error


    rmse = math.sqrt(rmse/count)
    avg = avg/count

    print(i)
    print("count", count)
    print("rmse:", rmse)
    print("avg:", avg)
    print("min:", min)
    print("max:", max)