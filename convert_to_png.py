    # convert a .png image file to a .bmp image file using PIL
from PIL import Image



for i in range(0,9):

    file_in = "/local_home/daniel/sparse_pig/liver1/left_normal_{}.png".format(i)
    img = Image.open(file_in)

    r, g, b, a = img.split()
    img = Image.merge("RGB", (r, g, b))

    file_out = "/local_home/daniel/sparse_pig/liver1/left_normal_{}.bmp".format(i)
    img.save(file_out)

    file_in = "/local_home/daniel/sparse_pig/liver1/right_normal_{}.png".format(i)
    img = Image.open(file_in)


    r, g, b, a = img.split()
    img = Image.merge("RGB", (r, g, b))

    file_out = "/local_home/daniel/sparse_pig/liver1/right_normal_{}.bmp".format(i)
    img.save(file_out)