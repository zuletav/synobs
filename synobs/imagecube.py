import os
import warnings
import numpy as np
from astropy.io import fits
import scipy.constants as sc
import matplotlib.pyplot as plt

__all__ = ['imagecube']
warnings.filterwarnings('ignore')


class imagecube(object):
    """
    Base class containing all the FITS data. Must be a 3D cube containing two
    spatial and one velocity axis for and spectral shifting. A 2D 'cube' can
    be used to make the most of the deprojection routines. These can easily be
    made from CASA using the ``exportfits()`` command.

    Args:
        path (str): Relative path to the FITS cube.
        FOV (Optional[float]): Clip the image cube down to a specific
            field-of-view spanning a range ``FOV``, where ``FOV`` is in
            [arcsec].
        velocity_range (Optional[tuple]): A tuple of minimum and maximum
            velocities to clip the velocity range to.
        verbose (Optional[bool]): Whether to print out warning messages.
        primary_beam (Optional[str]): Path to the primary beam as a FITS file
            to apply the correction.
        bunit (Optional[str]): If no `bunit` header keyword is found, use this
            value, e.g., 'Jy/beam'.
        pixel_scale (Optional[float]): If no axis information is found in the
            header, use this value for the pixel scaling in [arcsec], assuming
            an image centered on 0.0".
    """

    frequency_units = {'GHz': 1e9, 'MHz': 1e6, 'kHz': 1e3, 'Hz': 1e0}
    velocity_units = {'km/s': 1e3, 'm/s': 1e0}

    def __init__(self, path, FOV=None, velocity_range=None, verbose=True,
                 primary_beam=None, bunit=None, pixel_scale=None):

        # Default parameters for user-defined values.

        self._user_bunit = bunit
        self._user_pixel_scale = pixel_scale
        if self._user_pixel_scale is not None:
            self._user_pixel_scale /= 3600.0

        # Read in the FITS data.

        self._read_FITS(path)
        self.verbose = verbose

        # Primary beam correction.

        self._pb_corrected = False
        if primary_beam is not None:
            self.correct_PB(primary_beam)

        # Cut down to a specific field of view.

        if FOV is not None:
            self._clip_cube_spatial(FOV/2.0)
        if velocity_range is not None:
            self._clip_cube_velocity(*velocity_range)
        if self.data.ndim == 3:
            self._velax_offset = self._calculate_symmetric_velocity_axis()
        if self.data.ndim != 3 and self.verbose:
            print("WARNING: Provided cube is only 2D. Shifting not available.")

    # -- Spectral Axis Manipulation -- #

    def velocity_to_restframe_frequency(self, velax=None, vlsr=0.0):
        """Return restframe frequency [Hz] of the given velocity [m/s]."""
        velax = self.velax if velax is None else np.squeeze(velax)
        return self.nu0 * (1. - (velax - vlsr) / 2.998e8)

    def restframe_frequency_to_velocity(self, nu, vlsr=0.0):
        """Return velocity [m/s] of the given restframe frequency [Hz]."""
        return 2.998e8 * (1. - nu / self.nu0) + vlsr

    def spectral_resolution(self, dV=None):
        """Convert velocity resolution in [m/s] to [Hz]."""
        dV = dV if dV is not None else self.chan
        nu = self.velocity_to_restframe_frequency(velax=[-dV, 0.0, dV])
        return np.mean([abs(nu[1] - nu[0]), abs(nu[2] - nu[1])])

    def velocity_resolution(self, dnu):
        """Convert spectral resolution in [Hz] to [m/s]."""
        v0 = self.restframe_frequency_to_velocity(self.nu0)
        v1 = self.restframe_frequency_to_velocity(self.nu0 + dnu)
        vA = max(v0, v1) - min(v0, v1)
        v1 = self.restframe_frequency_to_velocity(self.nu0 - dnu)
        vB = max(v0, v1) - min(v0, v1)
        return np.mean([vA, vB])

    # -- Masking Functions -- #

    def keplerian_mask(self, inc, PA, dist, mstar, vlsr, x0=0.0, y0=0.0,
                       z0=0.0, psi=1.0, r_cavity=None, r_taper=None,
                       q_taper=None, dV0=300.0, dVq=-0.5, r_min=0.0, r_max=4.0,
                       nbeams=None, tolerance=0.01, restfreqs=None,
                       max_dz0=0.2, return_type='float'):
        """
        Generate a make based on a Keplerian velocity model. Original code from
        ``https://github.com/richteague/keplerian_mask``. Unlike with the
        original code, the mask will be built on the same cube grid as the
        attached data. Multiple lines can be considered at once by providing a
        list of the rest frequencies of the line.

        Unlike other functions, this does not accept ``z_func``.

        Args:
            inc (float): Inclination of the disk in [deg].
            PA (float): Position angle of the disk, measured Eastwards to the
                red-shifted major axis from North in [deg].
            dist (float): Distance to source in [pc].
            mstar (float): Stellar mass in [Msun].
            vlsr (float): Systemic velocity in [m/s].
            x0 (Optional[float]): Source right ascension offset in [arcsec].
            y0 (Optional[float]): Source declination offset in [arcsec].
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            r_cavity (Optional[float]): Edge of the inner cavity for the
                emission surface in [arcsec].
            r_taper (Optional[float]): Characteristic radius in [arcsec] of the
                exponential taper to the emission surface.
            q_taper (Optional[float]): Exponent of the exponential taper of the
                emission surface.
            dV0 (Optional[float]): Line Doppler width at 1" in [m/s].
            dVq (Optional[float]): Powerlaw exponent for the Doppler-width
                dependence.
            r_min (Optional[float]): Inner radius to consider in [arcsec].
            r_max (Optional[float]): Outer radius to consider in [arcsec].
            nbeams (Optional[float]): Size of convolution kernel to smooth the
                mask by.
            tolerance (Optional[float]): After smoothing, the limit used to
                decide if a pixel is masked or not. Lower values will include
                more pixels.
            restfreqs (Optional[list]): Rest frequency (or list of rest
                frequencies) in [Hz] to allow for multiple (hyper-)fine
                components.
            max_dz0 (Optional[float]): The maximum step size between different
                ``z0`` values used for the different emission heights.
            return_type (Optional[str]): The value type used for the returned
                mask, the default is ``'float'``.

        Returns:
            ndarry:
                The Keplerian mask with the desired value type.
        """

        # Define the radial line width profile.

        def dV(r):
            return dV0 * r**dVq

        # Calculate the different heights that we'll have to use. For this we
        # use steps of `max_dz0`. The use of `linspace` at the end is to ensure
        # that the steps in z0 are equal.

        if z0 != 0.0:
            z0s = np.arange(0.0, z0, max_dz0)
            z0s = np.append(z0s, z0) if z0s[-1] != z0 else z0s
            z0s = np.concatenate([-z0s[1:][::-1], z0s])
            z0s = np.linspace(z0s[0], z0s[-1], z0s.size)
        else:
            z0s = np.zeros(1)

        # For each line center we need to loop through the different emission
        # heights. Each mask is where the line center, ``v_kep``, is within the
        # local linewidth, ``dV``, and half a channel. We then collapse all the
        # masks down to a single mask.

        masks = []
        for _z0 in z0s:
            for restfreq in np.atleast_1d(restfreqs):
                offset = self.restframe_frequency_to_velocity(restfreq)
                rvals = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA,
                                         z0=_z0, psi=psi, r_cavity=r_cavity,
                                         r_taper=r_taper, q_taper=q_taper)[0]
                v_kep = self.keplerian(x0=x0, y0=y0, inc=inc, PA=PA,
                                       mstar=mstar, dist=dist,
                                       vlsr=vlsr+offset, z0=_z0,
                                       psi=psi, r_cavity=r_cavity,
                                       r_taper=r_taper, q_taper=q_taper,
                                       r_min=r_min, r_max=r_max)
                mask = abs(self.velax[:, None, None] - v_kep)
                masks += [mask < dV(rvals) + self.chan]
        mask = np.any(masks, axis=0).astype('float')
        assert mask.shape == self.data.shape, "wrong mask shape"

        # Apply smoothing to the mask to broaden or soften the edges. Anything
        # that results in a value above ``tolerance`` is assumed to be within
        # the mask.

        if nbeams:
            mask = self.convolve_with_beam(mask, scale=float(nbeams))
        return np.where(mask >= tolerance, 1.0, 0.0).astype(return_type)

    def _string_to_Hz(self, string):
        """
        Convert a string to a frequency in [Hz].
        """
        if isinstance(string, float):
            return string
        if isinstance(string, int):
            return string
        factor = {'GHz': 1e9, 'MHz': 1e6, 'kHz': 1e3, 'Hz': 1e0}
        for key in ['GHz', 'MHz', 'kHz', 'Hz']:
            if key in string:
                return float(string.replace(key, '')) * factor[key]

    # -- FITS I/O -- #

    def _read_FITS(self, path):
        """Reads the data from the FITS file."""

        # File names.
        self.path = os.path.expanduser(path)
        self.fname = self.path.split('/')[-1]

        # FITS data.
        self.header = fits.getheader(path)
        self.data = np.squeeze(fits.getdata(self.path))
        self.data = np.where(np.isfinite(self.data), self.data, 0.0)
        try:
            self.bunit = self.header['bunit']
        except KeyError:
            if self._user_bunit is not None:
                self.bunit = self._user_bunit
            else:
                print("WARNING: Not `bunit` header keyword found.")
                self.bunit = input("\t Enter brightness unit: ")

        # Position axes.
        self.xaxis = self._readpositionaxis(a=1)
        self.yaxis = self._readpositionaxis(a=2)
        self.dpix = np.mean([abs(np.diff(self.xaxis))])
        self.nxpix = self.xaxis.size
        self.nypix = self.yaxis.size

        # Spectral axis.
        self.nu0 = self._readrestfreq()
        try:
            self.velax = self._readvelocityaxis()
            if self.velax.size > 1:
                self.chan = np.mean(np.diff(self.velax))
            else:
                self.chan = np.nan
            self.freqax = self._readfrequencyaxis()
            if self.chan < 0.0:
                self.data = self.data[::-1]
                self.velax = self.velax[::-1]
                self.freqax = self.freqax[::-1]
                self.chan *= -1.0
        except KeyError:
            self.velax = None
            self.chan = None
            self.freqax = None
        try:
            self.channels = np.arange(self.velax.size)
        except AttributeError:
            self.channels = [0]

        # Check that the data is saved such that increasing indices in x are
        # decreasing in offset counter to the yaxis.
        if np.diff(self.xaxis).mean() > 0.0:
            self.xaxis = self.xaxis[::-1]
            self.data = self.data[:, ::-1]

        # Beam.
        self._read_beam()

    def _read_beam(self):
        """Reads the beam properties from the header."""
        try:
            if self.header.get('CASAMBM', False):
                beam = fits.open(self.path)[1].data
                beam = np.median([b[:3] for b in beam.view()], axis=0)
                self.bmaj, self.bmin, self.bpa = beam
            else:
                self.bmaj = self.header['bmaj'] * 3600.
                self.bmin = self.header['bmin'] * 3600.
                self.bpa = self.header['bpa']
            self.beamarea_arcsec = self._calculate_beam_area_arcsec()
            self.beamarea_str = self._calculate_beam_area_str()
        except Exception:
            print("WARNING: No beam values found. Assuming pixel as beam.")
            self.bmaj = self.dpix
            self.bmin = self.dpix
            self.bpa = 0.0
            self.beamarea_arcsec = self.dpix**2.0
            self.beamarea_str = np.radians(self.dpix / 3600.)**2.0
        self.bpa %= 180.0

    def print_beam(self):
        """Print the beam properties."""
        print('{:.2f}" x {:.2f}" at {:.1f} deg'.format(*self.beam))

    @property
    def beam(self):
        return self.bmaj, self.bmin, self.bpa

    @property
    def beams_per_pix(self):
        """Number of beams per pixel."""
        return self.dpix**2.0 / self.beamarea_arcsec

    @property
    def pix_per_beam(self):
        """Number of pixels in a beam."""
        return self.beamarea_arcsec / self.dpix**2.0

    @property
    def FOV(self):
        """Field of view."""
        return self.xaxis.max() - self.xaxis.min()

    def _clip_cube_velocity(self, v_min=None, v_max=None):
        """Clip the cube to within ``vmin`` and ``vmax``."""
        v_min = self.velax[0] if v_min is None else v_min
        v_max = self.velax[-1] if v_max is None else v_max
        i = abs(self.velax - v_min).argmin()
        i += 1 if self.velax[i] < v_min else 0
        j = abs(self.velax - v_max).argmin()
        j -= 1 if self.velax[j] > v_max else 0
        self.velax = self.velax[i:j+1]
        self.data = self.data[i:j+1]

    def _clip_cube_spatial(self, radius):
        """Clip the cube plus or minus clip arcseconds from the origin."""
        if radius > min(self.xaxis.max(), self.yaxis.max()):
            if self.verbose:
                print('WARNING: FOV = {:.1f}" larger than '.format(radius * 2)
                      + 'FOV of cube: {:.1f}".'.format(self.xaxis.max() * 2))
        else:
            xa = abs(self.xaxis - radius).argmin()
            if self.xaxis[xa] < radius:
                xa -= 1
            xb = abs(self.xaxis + radius).argmin()
            if -self.xaxis[xb] < radius:
                xb += 1
            xb += 1
            ya = abs(self.yaxis + radius).argmin()
            if -self.yaxis[ya] < radius:
                ya -= 1
            yb = abs(self.yaxis - radius).argmin()
            if self.yaxis[yb] < radius:
                yb += 1
            yb += 1
            if self.data.ndim == 3:
                self.data = self.data[:, ya:yb, xa:xb]
            else:
                self.data = self.data[ya:yb, xa:xb]
            self.xaxis = self.xaxis[xa:xb]
            self.yaxis = self.yaxis[ya:yb]
            self.nxpix = self.xaxis.size
            self.nypix = self.yaxis.size

    def _readspectralaxis(self, a):
        """Returns the spectral axis in [Hz] or [m/s]."""
        a_len = self.header['naxis%d' % a]
        a_del = self.header['cdelt%d' % a]
        a_pix = self.header['crpix%d' % a]
        a_ref = self.header['crval%d' % a]
        return a_ref + (np.arange(a_len) - a_pix + 1.0) * a_del

    def _readpositionaxis(self, a=1):
        """Returns the position axis in [arcseconds]."""
        if a not in [1, 2]:
            raise ValueError("'a' must be in [1, 2].")
        try:
            a_len = self.header['naxis%d' % a]
            a_del = self.header['cdelt%d' % a]
            a_pix = self.header['crpix%d' % a]
        except KeyError:
            if self._user_pixel_scale is None:
                print('WARNING: No axis information found.')
                _input = input("\t Enter pixel scale size in [arcsec]: ")
                self._user_pixel_scale = float(_input) / 3600.0
            a_len = self.data.shape[-1] if a == 1 else self.data.shape[-2]
            if a == 1:
                a_del = -1.0 * self._user_pixel_scale
            else:
                a_del = 1.0 * self._user_pixel_scale
            a_pix = a_len / 2.0 + 0.5
        axis = 3600.0 * a_del * (np.arange(a_len) - 0.5 * (a_len - 1.0))
        return axis

    def _readrestfreq(self):
        """Read the rest frequency."""
        try:
            nu = self.header['restfreq']
        except KeyError:
            try:
                nu = self.header['restfrq']
            except KeyError:
                try:
                    nu = self.header['crval3']
                except KeyError:
                    nu = np.nan
        return nu

    def _readvelocityaxis(self):
        """Wrapper for _velocityaxis and _spectralaxis."""
        a = 4 if 'stokes' in self.header['ctype3'].lower() else 3
        if 'freq' in self.header['ctype%d' % a].lower():
            specax = self._readspectralaxis(a)
            velax = (self.nu0 - specax) * sc.c
            velax /= self.nu0
        else:
            velax = self._readspectralaxis(a)
        return velax

    def _readfrequencyaxis(self):
        """Returns the frequency axis in [Hz]."""
        a = 4 if 'stokes' in self.header['ctype3'].lower() else 3
        if 'freq' in self.header['ctype3'].lower():
            return self._readspectralaxis(a)
        return self._readrestfreq() * (1.0 - self._readvelocityaxis() / sc.c)

    def _calculate_symmetric_velocity_axis(self):
        """Returns a symmetric velocity axis for decorrelation functions."""
        try:
            velax_symmetric = np.arange(self.velax.size).astype('float')
        except AttributeError:
            return np.array([0.0])
        velax_symmetric -= velax_symmetric.max() / 2
        if abs(velax_symmetric).min() > 0.0:
            velax_symmetric -= abs(velax_symmetric).min()
        if abs(velax_symmetric[0]) < abs(velax_symmetric[-1]):
            velax_symmetric = velax_symmetric[:-1]
        elif abs(velax_symmetric[0]) > abs(velax_symmetric[-1]):
            velax_symmetric = velax_symmetric[1:]
        velax_symmetric *= self.chan
        return velax_symmetric

    def frequency(self, vlsr=0.0, unit='GHz'):
        """
        A `velocity_to_restframe_frequency` wrapper with unit conversion.

        Args:
            vlsr (optional[float]): Sytemic velocity in [m/s].
            unit (optional[str]): Unit for the output axis.

        Returns:
            1D array of frequency values.
        """
        return self.frequency_offset(nu0=0.0, vlsr=vlsr, unit=unit)

    def frequency_offset(self, nu0=None, vlsr=0.0, unit='MHz'):
        """
        Return the frequency offset relative to `nu0` for easier plotting.

        Args:
            nu0 (optional[float]): Reference restframe frequency in [Hz].
            vlsr (optional[float]): Sytemic velocity in [m/s].
            unit (optional[str]): Unit for the output axis.

        Returns:
            1D array of frequency values.
        """
        nu0 = self.nu0 if nu0 is None else nu0
        nu = self.velocity_to_restframe_frequency(vlsr=vlsr)
        return (nu - nu0) / imagecube.frequency_units[unit]

    # -- Unit Conversions -- #

    def jybeam_to_Tb_RJ(self, data=None, nu=None):
        """[Jy/beam] to [K] conversion using Rayleigh-Jeans approximation."""
        nu = self.nu0 if nu is None else nu
        data = self.data if data is None else data
        jy2k = 1e-26 * sc.c**2 / nu**2 / 2. / sc.k
        return jy2k * data / self._calculate_beam_area_str()

    def jybeam_to_Tb(self, data=None, nu=None):
        """[Jy/beam] to [K] conversion using the full Planck law."""
        nu = self.nu0 if nu is None else nu
        data = self.data if data is None else data
        Tb = 1e-26 * abs(data) / self._calculate_beam_area_str()
        Tb = 2.0 * sc.h * nu**3 / Tb / sc.c**2
        Tb = sc.h * nu / sc.k / np.log(Tb + 1.0)
        return np.where(data >= 0.0, Tb, -Tb)

    def Tb_to_jybeam_RJ(self, data=None, nu=None):
        """[K] to [Jy/beam] conversion using Rayleigh-Jeans approxmation."""
        nu = self.nu0 if nu is None else nu
        data = self.data if data is None else data
        jy2k = 1e-26 * sc.c**2 / nu**2 / 2. / sc.k
        return data * self._calculate_beam_area_str() / jy2k

    def Tb_to_jybeam(self, data=None, nu=None):
        """[K] to [Jy/beam] conversion using the full Planck law."""
        nu = self.nu0 if nu is None else nu
        data = self.data if data is None else data
        Fnu = 2. * sc.h * nu**3 / sc.c**2
        Fnu /= np.exp(sc.h * nu / sc.k / abs(data)) - 1.0
        Fnu *= self._calculate_beam_area_str() / 1e-26
        return np.where(data >= 0.0, Fnu, -Fnu)

    def _calculate_beam_area_arcsec(self):
        """Beam area in square arcseconds."""
        omega = self.bmin * self.bmaj
        if self.bmin == self.dpix and self.bmaj == self.dpix:
            return omega
        return np.pi * omega / 4. / np.log(2.)

    def _calculate_beam_area_str(self):
        """Beam area in steradians."""
        omega = np.radians(self.bmin / 3600.)
        omega *= np.radians(self.bmaj / 3600.)
        if self.bmin == self.dpix and self.bmaj == self.dpix:
            print('is dpix?')
            return omega
        return np.pi * omega / 4. / np.log(2.)

    # -- Utilities -- #

    def estimate_RMS(self, N=5, r_in=0.0, r_out=1e10):
        """
        Estimate RMS of the cube based on first and last `N` channels and a
        circular area described by an inner and outer radius.

        Args:
            N (int): Number of edge channels to include.
            r_in (float): Inner edge of pixels to consider in [arcsec].
            r_out (float): Outer edge of pixels to consider in [arcsec].

        Returns:
            RMS (float): The RMS based on the requested pixel range.
        """
        r_dep = np.hypot(self.xaxis[None, :], self.yaxis[:, None])
        rmask = np.logical_and(r_dep >= r_in, r_dep <= r_out)
        if self.data.ndim == 3:
            rms = np.concatenate([self.data[:int(N)], self.data[-int(N):]])
            rms = np.where(rmask[None, :, :], rms, np.nan)
        elif self.data.ndim == 2:
            rms = np.where(rmask, self.data, np.nan)
        else:
            raise ValueError("Unknown data dimension.")
        return np.sqrt(np.nansum(rms**2) / np.sum(np.isfinite(rms)))

    def print_RMS(self, N=5, r_in=0.0, r_out=1e10):
        """Print the estimated RMS in Jy/beam and K (using RJ approx.)."""
        rms = self.estimate_RMS(N, r_in, r_out)
        rms_K = self.jybeam_to_Tb_RJ(rms)
        print('{:.2f} mJy/beam ({:.2f} K)'.format(rms * 1e3, rms_K))

    def correct_PB(self, path):
        """Correct for the primary beam given by ``path``."""
        if self._pb_corrected:
            raise ValueError("This data has already been PB corrected.")
        pb = np.squeeze(fits.getdata(path))
        if pb.shape == self.data.shape:
            self.data /= pb
        else:
            self.data /= pb[None, :, :]
        self._pb_corrected = True

    @property
    def rms(self):
        """RMS of the cube based on the first and last 5 channels."""
        return self.estimate_RMS(N=5)

    @property
    def extent(self):
        """Cube field of view for use with Matplotlib's ``imshow``."""
        return [self.xaxis[0], self.xaxis[-1], self.yaxis[0], self.yaxis[-1]]

    def convolve_with_beam(self, data, scale=1.0, circular=False,
                           convolve_kwargs=None):
        """
        Convolve the attached data with a 2D Gaussian kernel matching the
        synthesized beam. This can be scaled with ``scale``, or forced to be
        circular (taking the major axis as the radius of the beam).

        Args:
            data (ndarray): The data to convolve. Must be either 2D or 3D.
            scale (Optional[float]): Factor to scale the synthesized beam by.
            circular (Optional[bool]): Force a cicular kernel. If ``True1``,
                the kernel will adopt the scaled major axis of the beam to use
                as the radius.
            convolve_kwargs (Optional[dict]): Keyword arguments to pass to
                ``astropy.convolution.convolve``.

        Returns:
            ndarray:
                Data convolved with the requested kernel.
        """
        from astropy.convolution import convolve, Gaussian2DKernel
        kw = {} if convolve_kwargs is None else convolve_kwargs
        kw['preserve_nan'] = kw.pop('preserve_nan', True)
        kw['boundary'] = kw.pop('boundary', 'fill')
        bmaj = scale * self.bmaj / self.dpix / 2.355
        bmin = scale * self.bmin / self.dpix / 2.355
        kernel = Gaussian2DKernel(x_stddev=bmaj if circular else bmin,
                                  y_stddev=bmaj, theta=np.radians(self.bpa))
        if data.ndim == 3:
            from tqdm import trange
            convolved = []
            for cidx in trange(data.shape[0]):
                convolved += [convolve(data[cidx], kernel, **kw)]
            return np.squeeze(convolved)
        elif data.ndim == 2:
            return convolve(data, kernel, **kw)
        else:
            raise ValueError("`data` must be 2 or 3 dimensional.")

    # -- Plotting Functions -- #

    def plot_beam(self, ax, x0=0.1, y0=0.1, **kwargs):
        """
        Plot the sythensized beam on the provided axes.

        Args:
            ax (matplotlib axes instance): Axes to plot the FWHM.
            x0 (float): Relative x-location of the marker.
            y0 (float): Relative y-location of the marker.
            kwargs (dic): Additional kwargs for the style of the plotting.
        """
        from matplotlib.patches import Ellipse
        beam = Ellipse(ax.transLimits.inverted().transform((x0, y0)),
                       width=self.bmin, height=self.bmaj, angle=-self.bpa,
                       fill=False, hatch=kwargs.pop('hatch', '////////'),
                       lw=kwargs.pop('linewidth', kwargs.pop('lw', 1)),
                       color=kwargs.pop('color', kwargs.pop('c', 'k')),
                       zorder=kwargs.pop('zorder', 1000), **kwargs)
        ax.add_patch(beam)