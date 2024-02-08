import numpy as np

from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy import units as u
from astropy import constants as const

import scipy.constants as sc

fwhm_to_sigma = 1. / (8 * np.log(2))**0.5


from gofish import imagecube

class simulationcube(imagecube):

    def __init__(self, path, **kargs):
        super().__init__(path, **kargs)

    def synthetic_obs(self, bmaj=None, bmin=None, bpa=0.0, 
                      case=None, filename=None, overwrite=False, verbose=True):
        
        '''
        Generate synthetic observations by convolving the data spatially 
        and adding noise. The beam is assumed to be Gaussian. It will convert the units
        to 'Jy/beam' and save the data in a fits file.
        
        Args:
            bmaj (float): Beam major FWHM axis in arcsec.
            bmin (float): Beam minor FWHM axis in arcsec.
            bpa (float): Beam position angle in degrees.
            overwrite (bool): Overwrite the existing file.

        Returns:
            ndarray: Synthetic observations.
        '''

        data_syn = self.data.copy()

        bmin = bmaj if bmin is None else bmin
        beam = self.beam_kernel(bmaj, bmin, bpa)

        channel_spacing_new = np.abs(self.header['CDELT3'] * const.c.to('m/s').value / self.header['CRVAL3'])

        if verbose:
            print(f'Convolving the data with a {bmaj:.2f}" x {bmin:.2f}" beam')

        # List of frequencies associated to each channel
        nu = [self.header['CRVAL3'] + i * self.header['CDELT3'] for i in range(self.data.shape[0])] # in Hz
        conversion_factor = self.get_conversion_factor(bmaj, bmin, verbose)

        for i, nu_i in enumerate(nu):

            data_slice = self.data[i, :, :]

            data_syn[i, :, :] = convolve_fft(data_slice, beam, boundary='wrap')
            data_syn[i, :, :] *= conversion_factor # Jy/pixel to Jy/beam

            rms_requested, rms_old = self.get_rms(channel_spacing_new, nu_i, bmaj, bmin, bpa, case)
            #if i == 1:
            #    print(f'RMS requested: {rms_requested:.2e} Jy/beam, RMS old: {rms_old:.2e} Jy/beam')

            data_syn[i, :, :] = self.add_correlated_noise(data_syn[i, :, :], rms_requested, rms_old, beam)

        self.save_synthethic_obs(data_syn, bmaj, bmin, bpa, filename, overwrite)
        
        return None
    
    def beam_kernel(self, bmaj, bmin=None, bpa=0.0):

        cdelt2 = self.header['CDELT2']
        bmaj *= u.arcsec.to('deg') / cdelt2
        bmin = bmaj if bmin is None else bmin * u.arcsec.to('deg') / cdelt2

        bmaj *= fwhm_to_sigma
        bmin *= fwhm_to_sigma

        print(bmaj, bmin)

        return Gaussian2DKernel(bmaj, bmin, np.radians(90 + bpa))

    def add_correlated_noise(self, data, rms_new, rms_old, beam):

        noise = np.random.normal(size=data.shape)
        noise = convolve_fft(noise, beam, boundary='wrap')

        #print(f'rms_new: {rms_new:.2e}, rms_old: {rms_old:.2e}, std(noise): {np.std(noise):.2e}, std(data): {np.std(data):.2e}')
        #print(f'rms_new / std(noise): {rms_new / np.std(noise):.2e}, rms_new / std(data): {rms_new / np.std(data):.2e}')

        return data + noise * rms_new / np.std(noise)
    
    def get_conversion_factor(self, bmaj, bmin, verbose):
        if 'jy/pix' in self.header['bunit'].lower():

            dpix = self.dpix
            beam_area = np.pi * (bmaj * bmin) / (4.0 * np.log(2.0))

            conversion_factor = beam_area / dpix**2
            print(f'dpix is {dpix}, beam area is {beam_area}, conversion factor is {conversion_factor}')

            if verbose:
                print(f'Converting from {self.header["bunit"]} to Jy/beam')
                print(f'Conversion factor: {conversion_factor}')

            return conversion_factor

    def get_rms(self, channel_spacing_new, nu, bmaj, bmin, bpa, case=None):

        try:
            if case in ['MAPS', 'exoALMA']:
                #print(f'Selecting {case} noise levels')
                channel_spacing_old = 350.0 if case == 'MAPS' else 150.0 # in m/s
                T = 3.5 if case == 'MAPS' else 3.0
        except:
            AttributeError('Case not recognized. Please choose between MAPS or exoALMA')
        finally:
            rms_old = self.brightness_temperature_to_flux_density(T, nu, bmaj, bmin)
            rms = self.rms_new(channel_spacing_old, channel_spacing_new, rms_old)

        return rms, rms_old
    
    def brightness_temperature_to_flux_density(self, T, nu, bmaj, bmin):

        nu = self.nu0 if nu is None else nu
        # Calculate the beam area using the beam sigma
        beam_area = self.beam_area_str(bmaj, bmin)
        jy2k = 1e-26 * sc.c**2 / nu**2 / 2. / sc.k
        return T * beam_area / jy2k

    def rms_new(self, channel_spacing_old, channel_spacing_new, rms_old):
        
        return np.sqrt(channel_spacing_old/channel_spacing_new)*rms_old
    
    def beam_area_str(self, bmaj, bmin):
        """Beam area in steradians."""
        omega = np.radians(bmin / 3600.)
        omega *= np.radians(bmaj / 3600.)
        return np.pi * omega / 4. / np.log(2.)
    
    def save_synthethic_obs(self, data, bmaj=None, bmin=None, bpa=None, filename=None, overwrite=False):

        if filename is None:
            filename = self.path.replace('.fits', f'{bmaj}FWHM_synthehic.fits')
        
        hdu = fits.PrimaryHDU(data=data, header=self.header)
        hdu.data = data

        hdu.header['BMAJ'] = bmaj / 3600 # arcsec to deg
        hdu.header['BMIN'] = bmin / 3600
        hdu.header['BPA'] = bpa

        hdu.header['BUNIT'] = 'Jy/beam'

        hdu.writeto(filename, overwrite=overwrite)



