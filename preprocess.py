import pydicom
import glob
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import gzip
import numpy as np



def getImageArray(url):
	all_image_urls = glob.glob(url)
	all_images = np.empty((len(all_image_urls), 1024, 1024))
	for i, url in enumerate(all_image_urls):
		dcm_data = pydicom.read_file(url)
		im = dcm_data.pixel_array
		all_images[i] = im
	return all_images

	


train_images = getImageArray('train_images/*.dcm')
test_images = getImageArray('test_images/*.dcm')
