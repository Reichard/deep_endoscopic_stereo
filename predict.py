from dispnet import *

def main(num_epochs=500):
    network = DispNet()
    network.load('dispnet')

    batch = load_batch(700,'TRAIN','C')
    disp = network.predict(batch)
    disp.save('disp.png')

if __name__ == '__main__':
	main()
