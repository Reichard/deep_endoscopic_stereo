from dispnet import *

import os
from PIL import ImageChops
from PIL import ImageOps
from math import *

TRAINING_WIDTH = 960
TRAINING_HEIGHT = 540

NETWORK="dispnet_endo"
STATUS="train_status_endo"

batchsize = 24
epoch_count = 10000

def print_progress(value, max_value):
    width = 32
    fill = int(width*(value/(max_value)))
    empty = width - fill
    print( '\r['+ '#'*fill + ' '*empty  +']{}/{}'.format(value,max_value), end='')

def load_batches(base_path):
    print("loading dataset...")

    endo_mask_left = Image.open("endo_mask_left.png")
    endo_mask_right = Image.open("endo_mask_left.png")
    endo_mask_disp = ImageOps.grayscale(Image.open("endo_mask_disp.png"))

    left_images = []
    right_images = []
    disp_images = []

    directories = [os.path.join(base_path,o) for o in os.listdir(base_path) if os.path.isdir(os.path.join(base_path,o))]

    paths = []

    for directory in directories:
        for i in range(0, 1000, 4):
            left_filename = os.path.join(directory,"scene_left_normal{}.png").format(i)
            right_filename = os.path.join(directory,"scene_right_normal{}.png").format(i)
            disp_filename = os.path.join(directory,"scene_left_depth{}.disp").format(i)

            if(not os.path.isfile(left_filename)
                    or not os.path.isfile(right_filename)
                    or not os.path.isfile(disp_filename)):
                continue

            paths.append((left_filename,right_filename,disp_filename))

    for i in range(len(paths)):
        print_progress(i,len(paths))
        left_filename,right_filename,disp_filename = paths[i]
        left_image = Image.open(left_filename)
        right_image = Image.open(right_filename)
        #left_image = ImageChops.multiply(Image.open(left_filename),endo_mask_left)
        #right_image = ImageChops.multiply(Image.open(right_filename),endo_mask_right)

        disp_array = np.fromfile(disp_filename,np.float64).astype(np.float32).reshape((left_image.size[1],left_image.size[0]))
        disp_array *= TRAINING_WIDTH/left_image.size[0]
        #disp_image = ImageChops.multiply(Image.fromarray(disp_array.astype(np.uint8)),endo_mask_disp)
        disp_image = Image.fromarray(disp_array.astype(np.uint8))

        left_images.append(left_image)
        right_images.append(right_image)
        disp_images.append(disp_image)

    print_progress(len(paths),len(paths))

    B = batchsize
    batches = [make_batch(left_images[i:i+B],right_images[i:i+B],disp_images[i:i+B],None) for i in range(0,len(left_images),B)]

    val_batches = batches[:2]
    batches = batches[2:]

    #print("augment batches")
    #augmented_batches = []
    #for batch in batches:
    #    augmented_batches.append(augment_batch(batch))
    #    augmented_batches.append(augment_batch(batch))
    #    augmented_batches.append(augment_batch(batch))
    #batches.extend(augmented_batches)
    
    print("training batches: {}".format(len(batches)))
    print("validation batches: {}".format(len(val_batches)))

    return (batches,val_batches)

def main():
    #batches, val_batches = load_batches("/org/share/MediData/MedData/Simluation/dispnet/liver")
    batches, val_batches = load_batches("/media/daniel/output/")

    network = DispNet()

    if os.path.exists(NETWORK + ".npz"):
        network.load(NETWORK + ".npz")

    start_epoch = 0
    if os.path.exists(STATUS + '.npz'):
        start_epoch += np.load(STATUS + '.npz')['epoch']

    # Finally, launch the training loop.
    print("Starting training...")
    weights = [1,0,0,0,0,0]
    learning_rate = 0.00001
    network.set_weights(weights)
    network.set_learning_rate(learning_rate)

    # We iterate over epochs:
    for epoch in range(start_epoch, epoch_count):
        print('Epoch {}'.format(epoch+1))

        learning_rate = 0.00001 / ((epoch*0.2) + 1)
        #weights = [epoch+1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        
        print(' weights: {}'.format(weights))
        print(' learning_rate: {}'.format(learning_rate))

        network.set_learning_rate(learning_rate)
        network.set_weights(weights)

        train_err = 0
        train_batches = 0
        start_time = time.time()


        print('train')

        for batch_idx in range(0,len(batches)):
            print_progress(batch_idx,len(batches))
            train_err += network.train(batches[batch_idx])

        print_progress(len(batches), len(batches))
        print('')

        debug_image = network.debug(batches[0])
        debug_image.save(('/media/daniel/output/debug_endo/{}.png').format(epoch))

        val_err = 0
        print('validate')

        for batch_idx in range(0, len(val_batches)):
            print_progress(batch_idx, len(val_batches))
            val_err += network.validate(val_batches[batch_idx])
        print_progress(len(val_batches), len(val_batches))
        print('')

        debug_image = network.debug(val_batches[0])
        debug_image.save(('/media/daniel/output/debug_endo_val/{}.png').format(epoch))
        network.save(NETWORK)
        network.save(NETWORK + '.bak')
        np.savez(STATUS, epoch=epoch+1)
        np.savez(STATUS + '.bak', epoch=epoch+1)

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, epoch_count, time.time() - start_time))
        print("  weights: \t\t", weights)
        print("  training loss:\t\t{:.6f}".format(train_err / len(batches)))
        print("  validation loss:\t\t{:.6f}".format(val_err / len(val_batches)))

if __name__ == '__main__':
    main()
