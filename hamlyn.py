from dispnet import *
import scipy.ndimage
from PIL import ImageOps
import theano
import matplotlib

network = DispNet()
network.load('dispnet_hamlyn.npz')

BASE_PATH = "/org/share/MediData/MedData/Hamlyn Dataset/"
OUT_PATH = "hamlyn/out"

for i in range(0,20):
    left = Image.open("{}/left_rec_{}.png".format(BASE_PATH,i))
    right = Image.open("{}/right_rec_{}.png".format(BASE_PATH,i))

    start_time = time.time()
    d = network.predict_data(left,right)
    print("Time for prediction: {}".format(time.time() - start_time))

    scale = d.shape[0] / left.size[1]
    crop = int((d.shape[1] - left.size[0] * scale )/2)
    d = d[:,crop:-crop]

    zoom_factors = (left.size[1]/d.shape[0],left.size[0]/d.shape[1])
    disp = scipy.ndimage.interpolation.zoom(d, zoom_factors)

    data_file = open("{}/{}.disp".format(OUT_PATH,i),'w')
    data_file.write(str(disp.shape[1]) + " " + str(disp.shape[0]) + "\n")

    pred_image = Image.fromarray(disp.astype(np.uint8))

    for y in range(disp.shape[0]):
        for x in range(disp.shape[1]):
            data_file.write(str(disp[y][x]) + "\n")

    pred_image.save("{}/{}.png".format(OUT_PATH,i))

    dbg_img = network.debug(make_batch([left],[right]))
    dbg_img.save("hamlyn/debug/{}.png".format(i))