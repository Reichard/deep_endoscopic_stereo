from dispnet import *

network = DispNet()
network.load('dispnet.npz')

#left = Image.open("/org/share/MediData/MedData/Simluation/dispnet/finalpass/frames_finalpass/TEST/A/0000/left/0006.png")
#right = Image.open("/org/share/MediData/MedData/Simluation/dispnet/finalpass/frames_finalpass/TEST/A/0000/right/0006.png")

BASE_PATH = "/org/share/MediData/MedData/TMI Dataset/Stereo_SD_d_all/"

left = Image.open(BASE_PATH + "Stereo_SD_d_all_2/Stereo_SD_d_all_2_IMG_REC_left.bmp")
right = Image.open(BASE_PATH + "Stereo_SD_d_all_2/Stereo_SD_d_all_2_IMG_REC_right.bmp")

disp = network.predict_data(left,right)

data_file = open("disp.disp",'w')
data_file.write(str(disp.shape[0]) + " " + str(disp.shape[1]) + "\n")

for x in range(disp.shape[0]):
	for y in range(disp.shape[1]):
		data_file.write(str(disp[x][y]) + "\n")

network.predict(make_batch([left],[right])).save("disp.png")
