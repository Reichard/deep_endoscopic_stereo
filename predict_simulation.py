from dispnet import *
from PIL import Image

def main():
    network = DispNet()
    network.load('dispnet.npz')

    left = Image.open('/org/share/MediData/MedData/Simluation/dispnet/finalpass/frames_finalpass/TRAIN/C/0700/left/0006.png')
    right = Image.open('/org/share/MediData/MedData/Simluation/dispnet/finalpass/frames_finalpass/TRAIN/C/0700/right/0006.png')
    print( left.size )


    left = Image.open('/org/share/MediData/MedData/Simluation/BioSimCamera/deform_side_view/scene_left0000.bmp')
    right = Image.open('/org/share/MediData/MedData/Simluation/BioSimCamera/deform_side_view/scene_right0000.bmp')

    left = left.crop((0,0,640-100,480))
    right = right.crop((100,0,640,480))

    left = Image.open('/org/share/MediData/MedData/Simluation/Leber/zoom/scene_left0015.bmp')
    right = Image.open('/org/share/MediData/MedData/Simluation/Leber/zoom/scene_right0015.bmp')

    left = Image.open('./bg/left.png')
    right = Image.open('./bg/right.png')

    print( left.size )
    batch = make_batch([left],[right])

    disp = network.debug(batch)
    disp.save('disp_sim.png')

if __name__ == '__main__':
	main()
