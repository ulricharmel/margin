import numpy as np
from tqdm import tqdm
from margin import logger

import random
import os
from PIL import Image
import glob


def resize_and_save(filename, output_dir, ind, size):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    suffix = filename.split('/')[-1][0:5]+str(ind)+".jpg"
    image.save(os.path.join(output_dir, suffix))


def build_dataset(data_dir, output_dir, size, seed=230):
    """Split the images into train/val/test datasets
        Also resize the the images in the process
    Args:
        data_dir (dict)
        - folder with the dataset
        output_dir (str)
        - folder for the output dataset
        size (int)
        - resize to this shape
        seed (int)
        -shuffling seed to make experiment reproducible
    """

    assert os.path.isdir(data_dir), "Couldn't find the dataset at {}".format(data_dir)

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(data_dir)
    filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith('.jpg')]


    # Split the images in into 70% train, 20 % Val and 10% test
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(seed)
    filenames.sort()
    random.shuffle(filenames)

    split_1 = int(0.7 * len(filenames))
    split_2 = int(0.9* len(filenames))
    
    train_filenames = filenames[:split_1]
    val_filenames = filenames[split_1:split_2]
    test_filenames = filenames[split_2:]

    filenames = {'train': train_filenames,
                 'val': val_filenames,
                 'test': test_filenames}

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        logger.warning("Warning: output dir {} already exists".format(output_dir))

    # Preprocess train, val and test
    for split in ['train', 'val', 'test']:
        output_dir_split = os.path.join(output_dir, '{}'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            logger.warning("Warning: dir {} already exists".format(output_dir_split))

        logger.info("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))

        ii = 0 
        for filename in tqdm(filenames[split]):
            try:
                resize_and_save(filename, output_dir_split, ii, size=size)
                ii += 1
            except:
                logger.warning(f"Failed for image {filename}")
    

        # check and delete all failed images
        images = glob.glob(output_dir_split+"/*jpg")
        for imagePath in images:
            # initialize if the image should be deleted or not
            delete = False
            # try to load the image
            try:
                Image.open(imagePath).convert("RGB")
            except:
                delete = True
            # check to see if the image should be deleted
            if delete:
                # print("[INFO] deleting {}".format(imagePath))
                os.remove(imagePath)
            

    logger.info("Done building dataset")
