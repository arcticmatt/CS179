import sys
import os
import time
import numpy as np

from scipy.ndimage import imread
from scipy.misc import imsave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.transform import radon, iradon



def main():
    start = time.time()
    # Remember that image is stored row-major
    # So if we have a 900 x 1122 image,
    # shape of image data will be (1122, 900) 
    # (or 1122, 900, 3 with color info)

    # Read image (known to work with JPEG, PNG, TIFF)
    imgdata_fullcolor = imread(sys.argv[1])
    file_name = sys.argv[1].split('.')[0]
    # We assume we read in b/w "internal image" for our simulated CT
    # Since RGB components should all be the same, just use the red one
    image_data = imgdata_fullcolor[:,:,0]
    #print(image_data.shape)

    # Save textual output of image data
    # (Note: Save-out is printed as integer, should be fine)
    fname_text_input = file_name + "_0_input_numerical.txt"
    np.savetxt(fname_text_input, image_data, fmt='%d')




    # Save matplotlib rendering of image (false color)
    plt.figure(0)
    plt.imshow(image_data, interpolation='nearest')

    fname_mpl_input_fc = file_name + "_1_input_mpl_falsecolor.png"
    plt.savefig(fname_mpl_input_fc)



    # Save matplotlib rendering of image (grayscale)
    plt.figure(1)
    plt.imshow(image_data, interpolation='nearest', cmap = cm.Greys_r)

    fname_mpl_input_gray = file_name + "_2_input_mpl_grayscale.png"
    plt.savefig(fname_mpl_input_gray)




    # Forward radon transform
    n_angles = int(sys.argv[2])

    theta = np.arange(0, 180, 180.0/n_angles)
    fwd_radon_sinogram = radon(image_data, theta, circle=False)
    #print(fwd_radon_sinogram.shape)


    # Plot sinogram as image format 
    # (each column is a line of radiation measurement)
    plt.figure(2)
    plt.imshow(fwd_radon_sinogram, interpolation='nearest', cmap=cm.Greys_r)

    fname_sinogram_asimage = file_name + "_3_sinogram_as_image.png"
    plt.savefig(fname_sinogram_asimage)


    # Save sinogram data as space-delimited text
    # (This is what goes into the GPU-accelerated CT reconstruction)

    # (We transpose the output to match the conventions of the C program,
    # which takes each sinogram line row-wise)
    fname_sinogram_data = file_name + "_4_sinogram_data.txt"
    np.savetxt(fname_sinogram_data, np.transpose(fwd_radon_sinogram))



    # Reconstruct scan using the inverse Radon transform
    # Gives a good idea of what reconstructed image should look like
    image_reconstruct_bckwd = iradon(fwd_radon_sinogram, theta=theta, circle=False)
    #print(image_reconstruct_bckwd.shape)

    # Save reconstructed image
    plt.figure(3)
    plt.imshow(image_reconstruct_bckwd, interpolation='nearest', cmap=cm.Greys_r)

    fname_reconstruct = file_name + "_5_recon_output.png"
    plt.savefig(fname_reconstruct)
    print("\nSuccess! Run time to preprocess with {0} angles: {1} seconds".format(n_angles, time.time() - start))
    print('Maximum dimension size of "{0}": {1}'.format(sys.argv[1], max(image_data.shape)))





if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("python preprocess.py <image file> <# Angles>")
    else:
        main()
