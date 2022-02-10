# import the necessary packages
from imutils import paths
import argparse
import requests
import cv2
import os
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True,
	help="path to file containing image URLs")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory of images")
ap.add_argument("-p", "--prefix", default="img",
	help="prefix for filenames")
ap.add_argument("-c", "--clean", action="store_true",
	help="delete files that can't be open")

ap.add_argument("-nd", "--no_download", action="store_true",
	help="skip download")

ap.add_argument("-m", "--max", default=100, type=int,
	help="maximum number of images to download")

args = vars(ap.parse_args())
# grab the list of URLs from the input file, then initialize the
# total number of images downloaded thus far
rows = open(args["urls"]).read().strip().split("\n")
total = 0

if not os.path.exists(args['output']):
    os.mkdir(args['output'])

zfill = int(np.ceil(np.log10(args["max"])))

if not args["no_download"]:
    # loop the URLs
    for url in rows:
        # import pdb; pdb.set_trace()
        p = os.path.sep.join([args["output"], "{}-{}.jpg".format(
                args["prefix"], str(total).zfill(zfill))])
        try:
            # try to download the image
            r = requests.get(url, timeout=60)
            # save the image to disk
            f = open(p, "wb")
            f.write(r.content)
            f.close()
            # update the counter
            print("[INFO] downloaded: {}".format(p))
            total += 1
        # handle if any exceptions are thrown during the download process
        except:
            print("[INFO] error downloading {}...skipping".format(p))
        
        if total == args["max"]:
            break

if args["clean"]:
    # loop over the image paths we just downloaded
    for imagePath in paths.list_images(args["output"]):
        # initialize if the image should be deleted or not
        delete = False
        # try to load the image
        try:
            image = cv2.imread(imagePath)
            # if the image is `None` then we could not properly load it
            # from disk, so delete it
            if image is None:
                delete = True
            # if OpenCV cannot load the image then the image is likely
            # corrupt so we should delete it
        except:
            print("Except")
            delete = True
        # check to see if the image should be deleted
        if delete:
            print("[INFO] deleting {}".format(imagePath))
            os.remove(imagePath)
