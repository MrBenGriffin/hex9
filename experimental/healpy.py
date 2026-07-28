# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0


# import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
import astropy.units as u

if __name__ == '__main__':

    # Target coordinates (Sgr A*)
    # 18h 03m 32.14s and a declination of -30d 02m 06.96s,
    # Baade's RA 270.892°, dec −30.034
    ra = 270.892
    dec = 30.034
    hp = HEALPix(nside=256, order='nested', frame='icrs')
    # Convert spherical sky coordinates to healpy polar angles (theta, phi)
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)

    # Compute HEALPix Level 8 index (nside = 2^8 = 256) in NESTED scheme
    # nside = 1 << res
    # p = int(hp.ang2pix(nside, lng, lat, nest=True, lonlat=True))
    coord = SkyCoord(ra=266.41683 * u.deg, dec=-29.00781 * u.deg)
    pixel_index = hp.skycoord_to_healpix(coord)
    print(f"Baades: HEALPix Level 8 Pixel Index: {pixel_index}")
