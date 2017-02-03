from dispnet import *
import scipy.ndimage
from PIL import ImageOps

TRAINING_WIDTH = 960
TRAINING_HEIGHT = 540

network = DispNet()
network.load('dispnet_endo_latest.npz')
#network.load('dispnet_last.npz')

BASE_PATH = "/org/share/MediData/MedData/TMI Dataset/"
PACKAGE = "Stereo_SD_d_all"

def mask_image( img, mask ):
    assert(img.size == mask.size)
    for y in range(left.size[1]):
        for x in range(left.size[0]):
            if(mask.getpixel((x,y)) != (0,0,0,255)):
                img.putpixel((x,y), (0,0,0))

for i in range(1,36):
    left = Image.open("{}{}/{}_{}/{}_{}_IMG_REC_left.bmp".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i))
    right = Image.open("{}{}/{}_{}/{}_{}_IMG_REC_right.bmp".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i))

    left = ImageOps.autocontrast(left)
    right = ImageOps.autocontrast(right)

    left_mask = Image.open("{}{}/{}_{}/{}_{}_MASK_REC_left.png".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i))
    right_mask = Image.open("{}{}/{}_{}/{}_{}_MASK_REC_right.png".format(BASE_PATH,PACKAGE,PACKAGE,i,PACKAGE,i))

    mask_image(left,left_mask)
    mask_image(right,right_mask)

    disp_factor = left.size[0] / TRAINING_WIDTH

    d = network.predict_data(left,right)*disp_factor
    zoom_factors = (left.size[1]/d.shape[0],left.size[0]/d.shape[1])
    disp = scipy.ndimage.interpolation.zoom(d, zoom_factors)
    
    data_file = open("tmi/{}.disp".format(i),'w')
    data_file.write(str(disp.shape[1]) + " " + str(disp.shape[0]) + "\n")
    
    for y in range(disp.shape[0]):
        for x in range(disp.shape[1]):
            data_file.write(str(disp[y][x]) + "\n")
    
    pred_image = network.predict_image(left,right).resize(left.size,Image.BILINEAR)

    pred_image.save("tmi/{}.png".format(i))
