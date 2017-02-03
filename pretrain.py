from dispnet import *

batchsize = 10

def print_progress(value, max_value):
    width = 32
    fill = int(width*(value/max_value))
    empty = width - fill
    print( '\r['+ '#'*fill + ' '*empty  +']{}/{}'.format(value,max_value), end='')

def main():
    #dataset = Dataset('/media/data/dispnet')
    #dataset = Dataset('/org/share/MediData/MedData/Simluation/dispnet')
    dataset = Dataset('/media/daniel/dispnet')

    network = DispNet()

    if os.path.exists('dispnet.npz'):
        network.load('dispnet.npz')

    start_epoch = 0
    if os.path.exists('train_status.npz'):
        start_epoch += np.load('train_status.npz')['epoch']

    executor = ThreadPoolExecutor(max_workers=2)

    # Finally, launch the training loop.
    print("Starting training...")
    weights = [0,0,0,0,0,1]
    learning_rate = 0.000001

    # We iterate over epochs:
    for epoch in range(start_epoch, 500):
        print('Epoch {}'.format(epoch+1))

        for i in range(0,6):
            x = epoch-(5-i)*10-10
            weights[i] = max(0,1 - abs(x/20))

        if(epoch <= 0):
            weights = [0,0,0,0,0,1]
        elif(epoch >= 70):
            weights = [1,0,0,0,0,0]
        weights = [max(0.001,w) for w in weights]

        weights = [1,1,1,1,1,1]

        #if(epoch > 1000):
        #    learning_rate = 1/epoch

        network.set_weights(weights)
        network.set_learning_rate(learning_rate)

        train_err = 0
        train_batches = 0
        start_time = time.time()

        print(' weights: {}'.format(weights))
        print(' learning_rate: {}'.format(learning_rate))

        print('train')

        future = executor.submit(dataset.load_training_batch, 0, batchsize)
        batch = None
        num_batches = int(dataset.size/batchsize)
        for batch_idx in range(0,int(num_batches)):
            print_progress(batch_idx,num_batches)
            batch = future.result()
            future = executor.submit(dataset.load_training_batch, batch_idx, batchsize)
            #batch = dataset.load_training_batch(batch_idx,batchsize)
            if batch != None:
                train_err += network.train(batch)
                train_batches += 1
            else:
                print('no files for batch! ({})'.format(batch_idx))
        print_progress(num_batches, num_batches)
        print('')


        debug_image = network.debug(batch,0)
        debug_image.save(('out/debug_train/{}.png').format(epoch))

        debug_image = network.debug(batch,random.randrange(len(batch[0])))
        debug_image.save(('out/debug_train/{}_1.png').format(epoch))

        val_err = 0
        val_batches = 0.0001

        print('validate')

        future = executor.submit(dataset.load_validation_batch, 0, batchsize)
        for batch_idx in range(0, 10):
            print_progress(batch_idx, 10)
            batch = future.result()
            future = executor.submit(dataset.load_validation_batch, batch_idx, batchsize)
            #batch = dataset.load_validation_batch(batch_idx,batchsize)
            if batch != None: 
                val_err += network.validate(batch)
                val_batches += 1
        print_progress(10, 10)
        print('')

        batch = dataset.load_validation_batch(0,batchsize)
        debug_image = network.debug(batch)
        debug_image.save(('out/debug_val/{}.png').format(epoch))
        network.save('dispnet')
        network.save('dispnet.bak')
        np.savez('train_status', epoch=epoch+1)
        np.savez('train_status.bak', epoch=epoch+1)

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, 500, time.time() - start_time))
        print("  weights: \t\t", weights)
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

        print(train_err, train_batches, val_err, val_batches)

if __name__ == '__main__':
    main()
