from dispnet import *

import os
from PIL import ImageChops
from PIL import ImageOps
from math import *

from scipy import ndimage

from queue import *
from threading import *
import datetime

TRAINING_WIDTH = 960
TRAINING_HEIGHT = 540

NETWORK = "dispnet_sparse"
STATUS = "train_status_sparse"

TRAINING_DEBUG_DIR = "debug"
VALIDATION_DEBUG_DIR = "debug_validation"

BASE_PATH = "/local_home/daniel/sparse_pig/"

LOG_FILE = "logs/sparse.log"

BATCHSIZE = 24
epoch_count = 10000

theano.config.mode = "FAST_COMPILE"

def print_progress(value, max_value):
    width = 32
    fill = int(width * (value / max_value))
    empty = width - fill
    print('\r[' + '#' * fill + ' ' * empty + ']{}/{}           '.format(value, max_value), end='')


paths = []
val_paths = []

def load_paths():
    local_directories = sorted(os.listdir(BASE_PATH))

    directories = [os.path.join(BASE_PATH, o) for o in local_directories if
                   os.path.isdir(os.path.join(BASE_PATH, o))]

    for directory in directories:
        count = 0
        if "validation" in directory:
            max = 50000
        else:
            max = 50000
        for i in range(max):
            left_filename = os.path.join(directory, "left_normal_{}.png").format(i)
            right_filename = os.path.join(directory, "right_normal_{}.png").format(i)
            disp_filename = os.path.join(directory, "left_depth_{}.disp").format(i)

            if (not os.path.isfile(left_filename)
                or not os.path.isfile(right_filename)
                or not os.path.isfile(disp_filename)):
                continue

            count += 1

            if "validation" in directory:
                val_paths.append((left_filename, right_filename, disp_filename))
            else:
                paths.append((left_filename, right_filename, disp_filename))

        print(directory, count)

    if len(paths) == 0:
        print("No valid dataset found")
        print(BASE_PATH)
        exit(1)

def load_batches():
    print("loading dataset...")

    load_paths()

    endo_mask_left = None
    endo_mask_right = None
    endo_mask_disp = None

    #endo_mask_left = Image.open("endo_sparse_mask_left.png")
    #endo_mask_right = Image.open("endo_sparse_mask_left.png")
    #endo_mask_disp = ImageOps.grayscale(Image.open("endo_sparse_mask_disp.png"))

    left_images = []
    right_images = []
    disp_images = []
    disp_masks = []


    for i in range(len(paths)):
        print_progress(i, len(paths))
        left_filename, right_filename, disp_filename = paths[i]

        left_image = Image.open(left_filename)
        right_image = Image.open(right_filename)
        left_image.load()
        right_image.load()

        if(endo_mask_left != None):
            left_image = ImageChops.multiply(left_image, endo_mask_left)
        if(endo_mask_right != None):
            right_image = ImageChops.multiply(right_image, endo_mask_right)

        disp_array = np.fromfile(disp_filename, np.float64).astype(np.float32).reshape(
            (left_image.size[1], left_image.size[0]))

        disp_mask = np.copy(disp_array)
        disp_mask[disp_array > 0] = 1

        weight_array = ndimage.uniform_filter(disp_mask,8)
        disp_array = ndimage.uniform_filter(disp_array,8) / weight_array

        disp_image = Image.fromarray(disp_array.astype(np.uint8))

        disp_mask = Image.fromarray(disp_mask.astype(np.uint8))

        if(endo_mask_disp != None):
            disp_image = ImageChops.multiply(disp_image, endo_mask_disp)

        left_images.append(left_image.copy())
        right_images.append(right_image.copy())
        disp_images.append(disp_image)
        disp_masks.append(disp_mask)

    print_progress(len(paths), len(paths))

    B = BATCHSIZE
    batches = [make_batch(left_images[i:i + B], right_images[i:i + B], disp_images[i:i + B], disp_masks[i:i+B]) for i in
               range(0, len(left_images), B)]

    val_batches = batches[:2]
    batches = batches[2:]

    print("augment batches")
    augmented_batches = []
    #for batch in batches:
    #    augmented_batches.append(augment_batch(batch))
    #    augmented_batches.append(augment_batch(batch))
    #    augmented_batches.append(augment_batch(batch))
    #batches.extend(augmented_batches)

    print("training batches: {}".format(len(batches)))

    return (batches, val_batches)

batch_queue = Queue(10)
endo_mask_left = None
endo_mask_right = None
endo_mask_disp = None

def load_batch(i, val=False):
    if val:
        batch_paths = val_paths[i:i+BATCHSIZE]
    else:
        batch_paths = paths[i:i+BATCHSIZE]

    left_images = []
    right_images = []
    disp_images = []
    disp_masks = []

    for left_filename, right_filename, disp_filename in batch_paths:
        left_image = Image.open(left_filename)
        right_image = Image.open(right_filename)
        left_image.load()
        right_image.load()

        if(endo_mask_left != None and "liver" not in left_filename):
            left_image = ImageChops.multiply(left_image, endo_mask_left)
        if(endo_mask_right != None and "liver" not in left_filename):
            right_image = ImageChops.multiply(right_image, endo_mask_right)

        disp_load = np.fromfile(disp_filename, np.float64).astype(np.float32)

        if len(disp_load) != left_image.size[0]*left_image.size[1]:
            print("Disp and Image Resolution do not match!!")

        disp_array = disp_load.reshape(
            (left_image.size[1], left_image.size[0]))

        #TODO: fix this in the dispnet code
        #scale disps according to thumbnails scale
        #disp_array *= TRAINING_HEIGHT / left_image.size[1]

        disp_mask = np.copy(disp_array)
        disp_mask[disp_array > 0] = 1

        if "liver" not in left_filename :
            weight_array = ndimage.uniform_filter(disp_mask,8)
            disp_array = ndimage.uniform_filter(disp_array,8) / weight_array

        disp_image = Image.fromarray(disp_array.astype(np.uint8))

        disp_mask = Image.fromarray(disp_mask.astype(np.uint8))

        if(endo_mask_disp != None):
            disp_image = ImageChops.multiply(disp_image, endo_mask_disp)

        left_images.append(left_image.copy())
        right_images.append(right_image.copy())
        disp_images.append(disp_image)
        disp_masks.append(disp_mask)

    batch = make_batch(left_images,right_images,disp_images,disp_masks)

    #if "liver" in batch_paths[0][0]:
    #    if random.uniform(0.0,1.0) > 0.5:
    #        batch = augment_batch(batch)

    return batch

def batch_producer():
    global paths
    global endo_mask_disp
    global endo_mask_left
    global endo_mask_right

    load_paths()
    endo_mask_left = Image.open("endo_mask_rect_left.png")
    endo_mask_right = Image.open("endo_mask_rect_right.png")
    #endo_mask_disp = ImageOps.grayscale(Image.open("endo_sparse_mask_disp.png"))

    permutation = list(range(len(paths)//BATCHSIZE))
    random.shuffle(permutation)

    while(True):
        for i in range(len(paths)//BATCHSIZE):
            batch_queue.put(("train",i,load_batch(permutation[i]*BATCHSIZE)))

        for i in range(len(val_paths)//BATCHSIZE):
            batch_queue.put(("val",i,load_batch(i*BATCHSIZE,True)))

        random.shuffle(permutation)


def main():
    if not os.path.isdir(TRAINING_DEBUG_DIR):
        os.mkdir(TRAINING_DEBUG_DIR)

    if not os.path.isdir(VALIDATION_DEBUG_DIR):
        os.mkdir(VALIDATION_DEBUG_DIR)

    log_file = open(LOG_FILE,"a")
    print('', file=log_file)
    print(datetime.datetime.now(), file=log_file)
    print('', file=log_file)
    print("epoch learning_rate training_error validation_error", file=log_file)

    #batches, val_batches = load_batches()

    print("starting batch producer")
    producer = Thread(target=batch_producer)
    producer.daemon = True
    producer.start()

    network = DispNet(freeze_inner_layers = False)

    if os.path.exists(NETWORK + ".npz"):
        network.load(NETWORK + ".npz")
        print("loading network {} from file...".format(NETWORK))

    start_epoch = 0
    if os.path.exists(STATUS + '.npz'):
        start_epoch += np.load(STATUS + '.npz')['epoch']
        print("loading status {} from file".format(STATUS))

    # Finally, launch the training loop.
    print("Starting training...")
    weights = [1, 0, 0, 0, 0, 0]
    learning_rate = 0.0001
    network.set_weights(weights)
    network.set_learning_rate(learning_rate)

    print("get first batch")
    batch_type, batch_idx, batch = batch_queue.get()

    # We iterate over epochs:
    for epoch in range(start_epoch, epoch_count):
        print('Epoch {}'.format(epoch + 1))

        learning_rate = 0.000001 / ((epoch*0.2) + 1)
        #weights = [epoch + 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        print(' weights: {}'.format(weights))
        print(' learning_rate: {}'.format(learning_rate))
        network.set_learning_rate(learning_rate)
        network.set_weights(weights)

        val_err = 0
        val_batches = 0
        train_err = 0
        train_batches = 0
        start_time = time.time()
        batch_count = len(paths)//BATCHSIZE
        val_batch_count = len(val_paths)//BATCHSIZE

        while True:

            if batch_type=="train":

                #train
                if batch_idx == 0:
                    print("training")
                    debug_image = network.debug(batch)
                    debug_image.save(('{}/{}.png').format(TRAINING_DEBUG_DIR,epoch))


                train_err += network.train_outer(batch)
                train_batches += 1
                print_progress(batch_idx, batch_count)

                for i in range(1,4):
                    network.train_outer(augment_batch(batch))
                    print_progress(batch_idx + i/10, batch_count)


            elif batch_type=="val":

                #validate
                if(batch_idx == 0):
                    print_progress(batch_count, batch_count)
                    print('')
                    print("validation")
                    debug_image = network.debug(batch,7)
                    debug_image.save(('{}/{}.png').format(VALIDATION_DEBUG_DIR,epoch))

                val_err += network.validate_outer(batch)
                val_batches += 1

                print_progress(batch_idx, val_batch_count)


            batch_type, batch_idx, batch = batch_queue.get()

            if batch_type == "train" and batch_idx == 0:
                print_progress(val_batch_count, val_batch_count)
                print('')
                break


        network.save(NETWORK)
        network.save(NETWORK + '.bak')
        np.savez(STATUS, epoch=epoch + 1)
        np.savez(STATUS + '.bak', epoch=epoch + 1)

        if (epoch +1) % 10 == 0:
            network.save("current_network_epochs/dispnet_epoch_{}".format(epoch+1))
            np.savez("current_network_epochs/train_status_epoch_{}".format(epoch+1), epoch=epoch + 1)

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, epoch_count, time.time() - start_time))
        print("  weights: \t\t", weights)
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

        print("{} {} {} {}".format(epoch, learning_rate, train_err/train_batches, val_err/val_batches), file=log_file)

        log_file.flush()

if __name__ == '__main__':
    main()
