from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

fname = '../data/raw/dss_search.fits'
hdulist = fits.open(fname, memmap=True) #hdulist means that this is returning the headers
print("hdulist[0].data:" + "\n")
print(hdulist[0].data) #prints the info about the headers
head = fits.getheader(fname) #returns the header information as a dictionary
print("\n" + "head:" + "\n")
for key in head.keys():
    print(key) #prints the keys of the header dictionary
image = fits.getdata(fname) #returns the data as a numpy array
print("\n" + "image:" + "\n")
print(image)
print("\n" + "shape of image:" + "\n")
print(image.shape) #returns numpy array shape (number of rows, number of columns)

print("\n" + "date of image:" + "\n")
print(head['DATE-OBS'])

#display image

plt.rcParams['figure.figsize'] = [10, 10]
plt.imshow(image, cmap='gray', origin='lower') # origin='lower' to display image with origin (0,0) at the bottom-left corner
#can add vmin=##### for what color you want to be 'black' and vmax=##### for what color you want to be 'white'
plt.colorbar() # adds a colorbar to the side of the image
plt.show()



'''
open() arguments (from docs): 
'memmap=True' allows for larger files to be read without loading the entire file into memory at once. 
This is particularly useful for working with large FITS files, as it allows you to access and manipulate the data without 
consuming excessive memory resources.

'use_fsspec=True' is an argument that can be used when opening a FITS file to indicate that the file contains a 
"FITS Standard Specification" (FSS) header. used with remote and cloud-hosted files. Allows file paths to be opened.
It supports a range of remote and distributed storage backends, for example, you can access hubble space telescope images
located in the hubble's public amazon S3 bucket as follows:
'''    

#uri = "s3://stpubdata/hst/public/j8pu/j8pu0y010/j8pu0y010_drc.fits"

# Extract a 10-by-20 pixel cutout image
#with fits.open(uri, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:
    #cutout = hdul[1].section[10:20, 30:50]
    #print(cutout)
    
'''
In the example above, they used .section to ensure that only the necessary parts of the FITS image
are transferred from the server, rather than downloading the entire data array. This trick can significantly
speed up the code, or if we required small subets of large FITS files located on slow (remote) servers.
'''

