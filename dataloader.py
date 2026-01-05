import torch
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import os
# from typing import Any, Callable, Optional, Tuple
# from PIL import Image
import numpy as np


def get_dataloaders(args, require_max_min=False):
    train_loader, val_loader, test_loader = None, None, None
    
    # Har dataset
    if args.data in ['UCI', 'UniMib', 'OPPORTUNITY', 'PAMAP2', 'PAMAP22', 'WISDM', 'USC', 'MotionSense', 'MotionSense3', 'DSADS', 'HHAR', 'HHAR2']:
        if args.shot is None:
            data_feature_train = np.load(fr'data(har)/{args.data}/x_train.npy')
            data_label_train = np.load(fr'data(har)/{args.data}/y_train.npy')
            data_feature_test = np.load(fr'data(har)/{args.data}/x_test.npy')
            data_label_test = np.load(fr'data(har)/{args.data}/y_test.npy')
            data_feature_val = np.load(fr'data(har)/{args.data}/x_valid.npy')
            data_label_val = np.load(fr'data(har)/{args.data}/y_valid.npy')
        else:
            data_feature_train = np.load(fr'data(har)/{args.data}/{args.shot}/x_train.npy')
            data_label_train = np.load(fr'data(har)/{args.data}/{args.shot}/y_train.npy')
            data_feature_test = np.load(fr'data(har)/{args.data}/{args.shot}/x_test.npy')
            data_label_test = np.load(fr'data(har)/{args.data}/{args.shot}/y_test.npy')
            data_feature_val = np.load(fr'data(har)/{args.data}/{args.shot}/x_valid.npy')
            data_label_val = np.load(fr'data(har)/{args.data}/{args.shot}/y_valid.npy')
        if require_max_min:
                max_test, min_test = np.max(data_feature_test), np.min(data_feature_test)
        train_feature_tensor = torch.from_numpy(data_feature_train).float()
        train_label_tensor = torch.from_numpy(data_label_train).float().long()
        test_feature_tensor = torch.from_numpy(data_feature_test).float()
        test_label_tensor = torch.from_numpy(data_label_test).float().long()
        val_feature_tensor = torch.from_numpy(data_feature_val).float()
        val_label_tensor = torch.from_numpy(data_label_val).float().long()
        train_set = torch.utils.data.TensorDataset(train_feature_tensor, train_label_tensor)
        test_set = torch.utils.data.TensorDataset(test_feature_tensor, test_label_tensor)
        val_set = torch.utils.data.TensorDataset(val_feature_tensor, val_label_tensor)
    # elif args.data == 'Flower':
    #     data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(224),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    #     "test": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
    #                                transforms.ToTensor(),
    #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    #     image_path = 'EENN/flower_data'  # flower data set path
    #     assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    
    #     train_set = datasets.ImageFolder(root=os.path.join(image_path, "train"),
    #                                         transform=data_transform["train"])
    #     test_set = datasets.ImageFolder(root=os.path.join(image_path, "test"),
    #                                             transform=data_transform["test"])
    print('Number of training samples: ', len(train_set))
    print('Number of test samples: ', len(test_set))
    if args.use_valid:
        print('Number of valid samples: ', len(val_set))
    if args.use_valid:
        train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=False)
        test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=False)


    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=False)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=False)
            test_loader = val_loader
    if require_max_min:
        return train_loader, val_loader, test_loader, max_test, min_test
    return train_loader, val_loader, test_loader
