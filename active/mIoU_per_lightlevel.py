from distutils.command.build import build
import os
from shutil import copyfile

import numpy as np
from PIL import Image

import torch

road = [53.48,88.59,89.98,93.5,94.17,93.99,92.78,93.6,88.57,61.39]
sidewalk = [32.11,65.44,76.63,77.65,71.75,74.49,64.09,30.7,35.26]
building = [84.23,90.58,91.95,90.55,91.04,92.52,93.13,94.8,95.55,94.44]
wall = [9.82,34.05,37.7,72.07,83.58,95.63,98.35,81.37,53.15,18.19]
