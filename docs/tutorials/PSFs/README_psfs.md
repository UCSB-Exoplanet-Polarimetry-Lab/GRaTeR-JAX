A variety of point spread functions (PSFs) are available for use with GRaTer-JAX.

### F300M

Description of files:
- im_mask_rolls_F300M.npy = Binary masks indicating valid/unmasked pixels for each roll angle.
- psf_inds_rolls_F300M.npy = Indices that map each image to the appropriate PSF.
- psf_offsets_F300M.npy = Offset values for each PSF location relative to the image grid.
- psfs_F300M.npy = A stack of PSF images at various field locations or roll angles.

When to use these:

### F360M

Description of files:
- im_mask_rolls_F360M.npy = Binary masks indicating valid/unmasked pixels for each roll angle.
- psf_inds_rolls_F360M.npy = Indices that map each image to the appropriate PSF.
- psf_offsets_F360M.npy = Offset values for each PSF location relative to the image grid.
- psfs_F360M.npy = A stack of PSF images at various field locations or roll angles.

### GPI

Description of files:
- GPI_Hband_PSF.fits = an empirically generated PSF for Gemini Planet Imager H band observations


### When to use these:
When you are processing different static psfs like Gemini Planet Imager, you can define your own fits file in the same
format as GPI_Hband_PSF.fits to deal with them. However, if you need to handle dynamic psfs like JWST that handles spatially variable PSFs
and various roll angles, you should use the framework for the Winnie_PSF class.

