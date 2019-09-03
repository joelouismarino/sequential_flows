import os
import torch
import tarfile
import numpy as np
from .misc_data_util import transforms as trans
from .misc_data_util.url_save import save
from zipfile import ZipFile


def load_dataset(data_config):
    """
    Downloads and loads a variety of standard benchmark sequence datasets.
    Arguments:
        data_config (dict): dictionary containing data configuration arguments
    Returns:
        tuple of (train, val), each of which is a PyTorch dataset.
    """
    data_path = data_config['data_path'] # path to data directory
    if data_path is not None:
        assert os.path.exists(data_path), 'Data path not found.'

    dataset_name = data_config['dataset_name'] # the name of the dataset to load
    dataset_name = dataset_name.lower() # cast dataset_name to lower case
    train = val = None

    ############################################################################
    ## Video datasets
    ############################################################################
    if dataset_name == 'kth_actions':
        if not os.path.exists(os.path.join(data_path, 'kth_actions')):
            os.makedirs(os.path.join(data_path, 'kth_actions'))
        if not os.path.exists(os.path.join(data_path, 'kth_actions', 'train')):
            print('Downloading KTH Actions dataset...')
            actions = ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']
            for action in actions:
                print('Downloading ' + action + '...')
                save('http://www.nada.kth.se/cvap/actions/' + action + '.zip',
                    os.path.join(data_path, 'kth_actions', action + '.zip'))
                print('\n')
            print('Done.')

            print('Unzipping KTH Actions dataset...')
            for action in actions:
                print('Unzipping ' + action + '...')
                zip_ref = ZipFile(os.path.join(data_path, 'kth_actions', action + '.zip'), 'r')
                os.makedirs(os.path.join(data_path, 'kth_actions', action))
                zip_ref.extractall(os.path.join(data_path, 'kth_actions', action))
                zip_ref.close()
                os.remove(os.path.join(data_path, 'kth_actions', action + '.zip'))
            print('Done.')

            print('Processing KTH Actions dataset...')
            from misc_data_util.convert_kth_actions import convert
            convert(os.path.join(data_path, 'kth_actions'))
            import shutil
            for action in actions:
                shutil.rmtree(os.path.join(data_path, 'kth_actions', action))
            print('Done.')

        from .datasets import KTHActions
        train_transforms = []
        if data_config['img_hz_flip']:
            train_transforms.append(trans.RandomHorizontalFlip())
        transforms = [trans.Resize(data_config['img_size']),
                      trans.RandomSequenceCrop(data_config['sequence_length']),
                      trans.ImageToTensor(),
                      trans.ConcatSequence()]
        train_trans = trans.Compose(train_transforms + transforms)
        val_trans = trans.Compose(transforms)
        test_trans = trans.Compose([trans.Resize(data_config['img_size']),
                      trans.ImageToTensor(),
                      trans.ConcatSequence()])
        train = KTHActions(os.path.join(data_path, 'kth_actions', 'train'), train_trans)
        val   = KTHActions(os.path.join(data_path, 'kth_actions', 'val'), val_trans)
        test  = KTHActions(os.path.join(data_path, 'kth_actions', 'test'), test_trans)

    elif dataset_name == 'bair_robot_pushing':
        if not os.path.exists(os.path.join(data_path, 'bair_robot_pushing')):
            os.makedirs(os.path.join(data_path, 'bair_robot_pushing'))

        if not os.path.exists(os.path.join(data_path, 'bair_robot_pushing', 'train')):
            print('Downloading BAIR Robot Pushing dataset...')
            save('http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar',
                os.path.join(data_path, 'bair_robot_pushing', 'bair_robot_pushing_dataset_v0.tar'))
            print('Done.')

            print('Untarring BAIR Robot Pushing dataset...')
            tar = tarfile.open(os.path.join(data_path, 'bair_robot_pushing', 'bair_robot_pushing_dataset_v0.tar'))
            tar.extractall(os.path.join(data_path, 'bair_robot_pushing'))
            tar.close()
            os.remove(os.path.join(data_path, 'bair_robot_pushing', 'bair_robot_pushing_dataset_v0.tar'))
            print('Done.')

            print('Converting TF records...')
            from misc_data_util.convert_bair import convert
            convert(os.path.join(data_path, 'bair_robot_pushing'))
            import shutil
            shutil.rmtree(os.path.join(data_path, 'bair_robot_pushing', 'softmotion30_44k'))
            print('Done.')

        from .datasets import BAIRRobotPushing
        train_transforms = []
        if data_config['img_hz_flip']:
            train_transforms.append(trans.RandomHorizontalFlip())
        transforms = [trans.Resize(data_config['img_size']),
                      trans.RandomSequenceCrop(data_config['sequence_length']),
                      trans.ImageToTensor(),
                      trans.ConcatSequence()]
        train_trans = trans.Compose(train_transforms + transforms)
        test_trans = trans.Compose(transforms)
        train = BAIRRobotPushing(os.path.join(data_path, 'bair_robot_pushing', 'train'), train_trans)
        val  = BAIRRobotPushing(os.path.join(data_path, 'bair_robot_pushing', 'test'), test_trans)

    elif dataset_name == 'bair_robot':
        raise NotImplementedError

    elif dataset_name == 'moving_mnist':
        if not os.path.exists(os.path.join(data_path, 'moving_mnist')):
            os.makedirs(os.path.join(data_path, 'moving_mnist'))

        if not os.path.exists(os.path.join(data_path, 'moving_mnist', 'train')):
            print('Downloading Moving MNIST dataset...')
            save('http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy',
                 os.path.join(data_path, 'moving_mnist', 'mnist_test_seq.npy'))
            data = np.load(os.path.join(data_path, 'moving_mnist', 'mnist_test_seq.npy'))
            train_data = data[:,:9000,...]
            val_data = data[:,9000:,...]
            os.makedirs(os.path.join(data_path, 'moving_mnist', 'train'))
            np.save(os.path.join(data_path, 'moving_mnist', 'train', 'data.npy'), train_data)
            os.makedirs(os.path.join(data_path, 'moving_mnist', 'val'))
            np.save(os.path.join(data_path, 'moving_mnist', 'val', 'data.npy'), val_data)
            os.remove(os.path.join(data_path, 'moving_mnist', 'mnist_test_seq.npy'))

        from .datasets import MovingMNIST
        train_transforms = []
        transforms = [trans.RandomSequenceCrop(data_config['sequence_length']),
                      trans.ToTensor(),
                      trans.Normalize(0., 255.)]
        train_trans = trans.Compose(train_transforms + transforms)
        test_trans = trans.Compose(transforms)
        train = MovingMNIST(os.path.join(data_path, 'moving_mnist', 'train', 'data.npy'), train_trans)
        val  = MovingMNIST(os.path.join(data_path, 'moving_mnist', 'val',  'data.npy'), test_trans)

    elif dataset_name == 'stochastic_moving_mnist':
        from .datasets import StochasticMovingMNIST
        train_transforms = []
        transforms = [trans.RandomSequenceCrop(data_config['sequence_length']),
                      trans.ToTensor()]
        train_trans = trans.Compose(train_transforms + transforms)
        test_trans = trans.Compose(transforms)
        train = StochasticMovingMNIST(train_trans)
        val  = StochasticMovingMNIST(test_trans)
    else:
        raise Exception('Dataset name not found.')

    return train, val
