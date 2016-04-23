
import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def main():
    start = time.time()
	# Given output data from the CT reconstruction,
	# produce a grayscale image (PNG format)
    image_data = np.loadtxt(sys.argv[1])
    plt.figure(0)
    plt.imshow(image_data, interpolation='nearest', cmap = cm.Greys_r)

    fname_reconstruct_image = "output_image.png"
    plt.savefig(fname_reconstruct_image)
    print('\nSuccess! Output image stored in "{0}".'.format( fname_reconstruct_image))
    print("Run time to postprocess: {0} seconds.".format(time.time() - start))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("python postprocess.py <text file with CT reconstruction>")
    else:
        main()
