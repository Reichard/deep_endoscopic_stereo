from dispnet import *

network = DispNet()
network.load('dispnet.npz')

left = Image.open("/org/share/MediData/MedData/Simluation/dispnet/finalpass/frames_finalpass/TEST/A/0000/left/0006.png")
right = Image.open("/org/share/MediData/MedData/Simluation/dispnet/finalpass/frames_finalpass/TEST/A/0000/right/0006.png")

batch = make_batch([left],[right])
disp = network.predict(batch)

disp.save('disp.png')
