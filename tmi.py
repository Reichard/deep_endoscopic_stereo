from dispnet import *
import scipy.ndimage
from PIL import ImageOps
import theano
import matplotlib

theano.config.mode = "FAST_RUN"
theano.config.linker = "cvm"
theano.config.optimizer = "fast_compile"

TRAINING_WIDTH = 960
TRAINING_HEIGHT = 540


#network.load('dispnet_last.npz')

BASE_PATH = "/org/share/MediData/MedData/TMI Dataset/"
PACKAGE = "Stereo_SD_d_all"
OUT_PATH = "tmi/out"

# j = 5
# l = Image.open("{}{}/{}_{}/{}_{}_IMG_REC_left.bmp".format(BASE_PATH,PACKAGE,PACKAGE,j,PACKAGE,j))
# r = Image.open("{}{}/{}_{}/{}_{}_IMG_REC_right.bmp".format(BASE_PATH,PACKAGE,PACKAGE,j,PACKAGE,j))
# lhsv = matplotlib.colors.rgb_to_hsv(np.array(l,dtype=np.float)/255)
# rhsv = matplotlib.colors.rgb_to_hsv(np.array(r,dtype=np.float)/255)
#
#
# l.show()
# for y in range(l.size[1]):
#     for x in range(l.size[0]):
#         hsv_left = lhsv[y][x]
#         is_spec = hsv_left[2] > 0.9 and  hsv_left[1] < 0.3
#
#         if is_spec:
#             l.putpixel((x,y), 0)
# l.show()
# exit()

network = DispNet()
network.load('dispnet_tmi.npz')

def mask_image( img, mask, channels = 3 ):
    assert(img.size == mask.size)
    for y in range(left.size[1]):
        for x in range(left.size[0]):
            if(mask.getpixel((x,y)) != (0,0,0,255)):
                if(channels == 1):
                    img.putpixel((x,y), 0)
                else:
                    img.putpixel((x,y), (0,)*channels)

for i in range(1,36):
    left = Image.open("{}{}/{}_{}/{}_{}_IMG_REC_left.bmp".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i))
    right = Image.open("{}{}/{}_{}/{}_{}_IMG_REC_right.bmp".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i))

    #left = ImageOps.autocontrast(left)
    #right = ImageOps.autocontrast(right)

    left_mask = Image.open("{}{}/{}_{}/{}_{}_MASK_REC_left.png".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i))
    right_mask = Image.open("{}{}/{}_{}/{}_{}_MASK_REC_right.png".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i))



#    left_image_hsv = matplotlib.colors.rgb_to_hsv(np.array(left,dtype=np.float)/255)
#    for y in range(left.size[1]):
#        for x in range(left.size[0]):
#            hsv_left = left_image_hsv[y][x]
#            is_spec = hsv_left[2] > 0.9 and hsv_left[1] < 0.3
#            if is_spec:
#                left.putpixel((x,y), 0)

#    right_image_hsv = matplotlib.colors.rgb_to_hsv(np.array(right,dtype=np.float)/255)
#    for y in range(right.size[1]):
#       for x in range(right.size[0]):
#            hsv_right = right_image_hsv[y][x]
#            is_spec = hsv_right[2] > 0.9 and hsv_right[1] < 0.3
#            if is_spec:
#                right.putpixel((x,y), 0)

#    mask_image(left,left_mask)
#    mask_image(right,right_mask)

    #disp_factor = left.size[1] / TRAINING_HEIGHT

    start_time = time.time()

    d = network.predict_data(left,right)#*disp_factor

    print("Time for prediction: {}".format(time.time() - start_time))

    scale = d.shape[0] / left.size[1]

    crop = int((d.shape[1] - left.size[0] * scale )/2)

    d = d[:,crop:-crop]

    zoom_factors = (left.size[1]/d.shape[0],left.size[0]/d.shape[1])
    disp = scipy.ndimage.interpolation.zoom(d, zoom_factors)
    
    data_file = open("{}/{}.disp".format(OUT_PATH,i),'w')
    data_file.write(str(disp.shape[1]) + " " + str(disp.shape[0]) + "\n")

    #load evaluation mask
    eval_mask = Image.open("{}{}/{}_{}/{}_{}_MASK_REC_eval.png".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i))

    pred_image = Image.fromarray(disp.astype(np.uint8))

    left_image_hsv = matplotlib.colors.rgb_to_hsv(np.array(left,dtype=np.float)/255)

    for y in range(disp.shape[0]):
        for x in range(disp.shape[1]):
            hsv_left = left_image_hsv[y][x]
            is_spec = hsv_left[2] > 0.9 and hsv_left[1] < 0.3

            #rx = 2*abs(x-disp.shape[1]/2)/disp.shape[1]
            #ry = 2*abs(y-disp.shape[0]/2)/disp.shape[0]

            #dist = rx*rx + ry*ry

            if eval_mask.getpixel((x,y)) != 0 or is_spec: #dist > 1.1 or
                data_file.write(str(0) + "\n")
                pred_image.putpixel((x,y), 0)
            else:
                data_file.write(str(disp[y][x]) + "\n")

    dbg_img = network.debug(make_batch([left],[right]))
    dbg_img.save("tmi/debug/{}.png".format(i))
    
    pred_image.save("{}/{}.png".format(OUT_PATH,i))