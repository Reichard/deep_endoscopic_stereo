from dispnet import *
from PIL import Image

def main():
    network = DispNet()
    network.load('dispnet.npz')

    left = Image.open('./tier2/left/20.png')
    right = Image.open('./tier2/right/20.png')
    left = left.crop((600,21,1300,510))
    right = right.crop((600,21,1300,510))

    batch = make_batch([left],[right])

    disp = network.debug(batch)
    disp.save('disp.png')

if __name__ == '__main__':
	main()
