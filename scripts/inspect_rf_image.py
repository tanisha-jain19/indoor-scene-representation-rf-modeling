"""Portable version of the original image.py inspection utility."""
import argparse
import numpy as np
from PIL import Image

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('image');args=parser.parse_args()
    with Image.open(args.image) as image:array=np.asarray(image)
    print('Shape:',array.shape,'Minimum:',array.min(),'Maximum:',array.max())
