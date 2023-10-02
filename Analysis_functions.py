#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pf
import os
from astropy.modeling import models
from astropy import units as u
from specutils import Spectrum1D
from specutils.analysis import equivalent_width
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle
from scipy.special import voigt_profile
from scipy.interpolate import interp1d
import scipy.odr as so
import pandas as pd
import itertools
import lmfit as lf
import scipy.stats as ss
import bindensity as bind

def extraction(file_directory, blaze_directory, CCF_directory, order):
    '''
    Function to extract the important quantities from the FITS files for a given day of solar observations.
    Parameters
    ----------
    :param file_directory: string, name of directory containing the spectral FITS files for a given day of solar observations.
    :param blaze_directory: string, name of directory containing the blaze FITS files for a given day of solar observations.
    There will be one or two files depending on the number of modes of observation.
    :param CCF_directory: string, name of directory containing the CCF FITS files for a given day of solar observations.
    :param order: int, order of the Échelle spectrograph we want to use.
    Returns
    ----------
    :param total_lamda: nested array, containing list of wavelengths for each spectrum, for the given order.
    :param total_spctr: nested array, containing list of flux values for each spectrum, for the given order.
    :param total_norm_spctr: nested array, containing list of normalized flux values for each spectrum, for the given order.
    :param total_err: nested array, containing list of errors on the flux values for each spectrum, for the given order.
    :param total_norm_err: nested array, containing list of errors on the normalized flux values for each spectrum, 
    for the given order.
    :param total_SNR: nested array, containing the SNR value for each spectrum, for the given order.
    :param mode: nested array, containing the mode of observation for each spectrum.
    :param date: nested array, containing timestamp of each spectrum.
    :param total_RV: nested array, containing the RV values for each spectrum, obtained from the CCF.
    :param total_RV_err: nested array, containing the error on the RV values for each spectrum, obtained from the CCF.
    :param total_FWHM: nested array, containing the FWHM values for the CCF of each spectrum.
    :param total_FWHM_err: nested array, containing the error on the FWHM for the CCF of each spectrum.
    :param total_BIS_SPAN: nested array, containing the Bisector Span values for the CCF of each spectrum.
    :param total_BIS_SPAN_err: nested array, containing the error on the Bisector Span for the CCF of each spectrum.
    :param total_CONTRAST: nested array, containing the Contrast values for the CCF of each spectrum.
    :param total_CONTRAST_err: nested array, containing the error on the Contrast for the CCF of each spectrum.
    :param total_H2O: nested array, containing the integrated column density of H2O for each time of observation.
    :param total_H2O_err: nested array, containing the error on the integrated column density of H2O for each time of observation.
    :param total_O2: nested array, containing the integrated column density of O2 for each time of observation.
    :param total_O2_err: nested array, containing the error on the integrated column density of O2 for each time of observation.   
    :param total_CO2: nested array, containing the integrated column density of CO2 for each time of observation.
    :param total_CO2_err: nested array, containing the error on the integrated column density of CO2 for each time of observation. 
    :param total_AIRM: nested array, containing the airmass at the time of observation for each time of observation.
    '''
    
    #Initialize the arrays.
    #Contains the wavelength for the order of interest for the spectra.
    total_lamda = np.zeros((len(os.listdir(file_directory)), 4084))
    
    #Contains the flux and normalized values of the spectra of the order of interest.
    total_spctr = np.zeros((len(os.listdir(file_directory)), 4084))
    total_norm_spctr = np.zeros((len(os.listdir(file_directory)), 4084))
    
    #Contains the error on the flux and normalized flux values of the spectra.
    total_err = np.zeros((len(os.listdir(file_directory)), 4084))
    total_norm_err = np.zeros((len(os.listdir(file_directory)), 4084))
    
    #Contains the SNR value of the spectra.
    total_SNR = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the mode of the spectra.
    mode = np.zeros((len(os.listdir(file_directory))), dtype=str)
    
    #Contains the timestamp at which each spectrum was collected.
    date = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the RV values of the spectra obtained from the CCF.
    total_RV = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the error on the RV values of the spectra.
    total_RV_err = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the FWHM of the CCF of the spectra.
    total_FWHM = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the error on the FWHM of the CCF of the spectra.
    total_FWHM_err = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the Bisector span of the CCF of the spectra.
    total_BIS_SPAN = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the error on the Bisector span of the CCF of the spectra.
    total_BIS_SPAN_err = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the Contrast of the CCF of the spectra.
    total_CONTRAST = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the error on the Contrast of the CCF of the spectra.
    total_CONTRAST_err = np.zeros((len(os.listdir(file_directory))))

    #Contains the integrated column density of H2O, CO2 and O2 at the time of acquisition of the spectra.
    total_H2O = np.zeros((len(os.listdir(file_directory))))
    total_O2 = np.zeros((len(os.listdir(file_directory))))
    total_CO2 = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the error on the integrated column density of H2O, CO2 and O2 at the time of acquisition of the spectra.
    total_H2O_err = np.zeros((len(os.listdir(file_directory))))
    total_O2_err = np.zeros((len(os.listdir(file_directory))))
    total_CO2_err = np.zeros((len(os.listdir(file_directory))))

    #Contains the airmass at the time of observation of each spectrum.
    total_AIRM = np.zeros((len(os.listdir(file_directory))))
    
    #Defining the Sun's BB for reduction purposes.
    Sun_BB = models.BlackBody(temperature = 5778*u.K)

    #Initial mode analysis to determine the modes used in the data.
    for i in range(len(os.listdir(file_directory))):
        file = pf.open(file_directory+'/'+sorted(os.listdir(file_directory))[i])
        mode[i] = file[0].header['HIERARCH ESO INS MODE'][1]
    
    #Loop over all the files in the directory and populate the arrays.
    for i in range(len(os.listdir(file_directory))):
        #Opening the files in the directories.
        file = pf.open(file_directory+'/'+sorted(os.listdir(file_directory))[i])
        file_CCF = pf.open(CCF_directory +'/'+sorted(os.listdir(CCF_directory))[i])
        
        #Extracting the wavelength.
        total_lamda[i] = file[4].data[order]

        #Extracting the DLL for spectrum correction.
        file_DLL = file[6].data[order]
        
        #Extracting the timestamp of each spectra.
        date[i] = file[0].header['MJD-OBS']

        #Getting the error and value of the RV from the CCF files.
        total_RV[i] = file_CCF[0].header['HIERARCH ESO QC CCF RV']
        total_RV_err[i] = file_CCF[0].header['HIERARCH ESO QC CCF RV ERROR']
        
        #Getting the error and value of the FWHM.
        total_FWHM[i] = file_CCF[0].header['HIERARCH ESO QC CCF FWHM']
        total_FWHM_err[i] = file_CCF[0].header['HIERARCH ESO QC CCF FWHM ERROR']
        
        #Getting the error and value of the Bisector Span.
        total_BIS_SPAN[i] = file_CCF[0].header['HIERARCH ESO QC CCF BIS SPAN']
        total_BIS_SPAN_err[i] = file_CCF[0].header['HIERARCH ESO QC CCF BIS SPAN ERROR']

        #Getting the error and value of the Contrast.
        total_CONTRAST[i] = file_CCF[0].header['HIERARCH ESO QC CCF CONTRAST']
        total_CONTRAST_err[i] = file_CCF[0].header['HIERARCH ESO QC CCF CONTRAST ERROR']
        
        #Getting the airmass .
        total_AIRM[i] = (file[0].header['HIERARCH ESO TEL AIRM START'] + file[0].header['HIERARCH ESO TEL AIRM END'])/2
    
        #Getting the error and value of the integrated column density for H2O, O2, CO2.
        total_H2O[i] = file[0].header['HIERARCH ESO QC TELL H2O IWV']
        total_H2O_err[i] = file[0].header['HIERARCH ESO QC TELL H2O IWV ERR']
        total_O2[i] = file[0].header['HIERARCH ESO QC TELL O2 IWV']
        total_O2_err[i] = file[0].header['HIERARCH ESO QC TELL O2 IWV ERR']
        total_CO2[i] = file[0].header['HIERARCH ESO QC TELL CO2 IWV']
        total_CO2_err[i] = file[0].header['HIERARCH ESO QC TELL CO2 IWV ERR']
    
        #Retrieving the SNR of each spectra at the order of interest.
        total_SNR[i] = file[0].header['HIERARCH ESO QC ORDER'+str(order)+' SNR']
        
        #Getting the raw spectrum, that has bee Telluric-Corrected(TC).
        file_spctr = file[8].data[order]
        file_err = file[10].data[order]

        #Distinguish two cases depending on the number of modes of observation.
        #If there are two modes of observation.
        if np.sum(mode=='A') != len(mode) and np.sum(mode=='E')!= len(mode):
            #Looking through the Blaze directory to find the Blaze files for each observation mode.
            for j in range(len(os.listdir(blaze_directory))):
                file_blaze = pf.open(blaze_directory+'/'+os.listdir(blaze_directory)[j])
                if file_blaze[0].header['HIERARCH ESO INS MODE'][1] == 'A':
                    blaze_HA = file_blaze
                else:
                    blaze_HE = file_blaze
                         
            #Getting the Blaze spectra for each mode of observation.
            blaze_HA_spctr = blaze_HA[1].data[order]
            blaze_HE_spctr = blaze_HE[1].data[order]
            
            #Performing the Blaze correction -- the Blaze file used depends on the observation mode used.
            if mode[i] == 'A':
                BC_spctr = file_spctr/blaze_HA_spctr
                BC_err = file_err/blaze_HA_spctr
            else:
                BC_spctr = file_spctr/blaze_HE_spctr
                BC_err = file_err/blaze_HE_spctr

            #Removing the downward trend of the spectrum's continuum using DLL and the Black Body of the Sun.
            BC_DLL_spctr = BC_spctr/file_DLL
            BC_DLL_err = BC_err/file_DLL
            
            total_spctr[i] = BC_DLL_spctr/Sun_BB(total_lamda[i]*u.AA).value
            total_err[i] = BC_DLL_err/Sun_BB(total_lamda[i]*u.AA).value
            
            #Making the normalized spectra and error bars. -- re-make with Rassine
            total_norm_spctr[i] = total_spctr[i]/np.average(total_spctr[i], weights = 1/total_err[i]**2)
            total_norm_err[i] = total_err[i]/np.average(total_spctr[i], weights = 1/total_err[i]**2)
                    
        #If there is only one mode of observation.
        else:
            #Getting the blaze file.
            blaze = pf.open(blaze_directory+'/'+os.listdir(blaze_directory)[0])
            
            #Getting the blaze spectra.
            blaze_spctr = blaze[1].data[order]
            
            #Performing the Blaze correction.
            BC_spctr = file_spctr/blaze_spctr
            BC_err = file_err/blaze_spctr
            
            #Removing the trend of the spectrum's continuum using DLL and the Sun's Black Body spectrum.
            BC_DLL_spctr = BC_spctr/file_DLL
            BC_DLL_err = BC_err/file_DLL
            
            total_spctr[i] = BC_DLL_spctr/Sun_BB(total_lamda[i]*u.AA).value
            total_err[i] = BC_DLL_err/Sun_BB(total_lamda[i]*u.AA).value
            
            #Making the normalized spectra and error bars.
            total_norm_spctr[i] = total_spctr[i]/np.average(total_spctr[i], weights = 1/total_err[i]**2)
            total_norm_err[i] = total_err[i]/np.average(total_spctr[i], weights = 1/total_err[i]**2)

    return total_lamda, total_spctr, total_norm_spctr, total_err, total_norm_err, total_SNR, mode, date, total_RV, total_RV_err, total_FWHM, total_FWHM_err, total_BIS_SPAN, total_BIS_SPAN_err, total_CONTRAST, total_CONTRAST_err, total_H2O, total_H2O_err, total_O2, total_O2_err, total_CO2, total_CO2_err, total_AIRM



def segment_and_reduce(modes, SNR, L, RV, airmass, sigma_coeff_SNR, airmass_limit = 2, sigma_coeff_RV=20):
    '''
    Function to remove the data for the spectra with low SNR and separate
    the data based on the mode of observation used.
    Parameters
    ----------
    :param modes: nested array, containing the mode of observation for each spectrum.
    :param SNR: nested array, containing the SNR value for each spectrum.
    :param L: nested array, containing the quantity of interest for each spectrum (RV, flux, ...).
    See the possible quantities output by the extraction function.
    :param RV: nested array, containing the RV values from the CCF for each observation.
    :param airmass: array, containing the airmass values for each time of observation.
    :param sigma_coeff_SNR: int, acceptance parameter used in our SNR clipping method.
    :param airmass_limit: float, limiting airmass above which observations are discarded.
    :param sigma_coeff_RV: int, used in the RV sigma clipping method described below.
    Returns
    ----------
    :param L_HA: nested array, containing the data observed with the observation mode HA,
    with the low SNR and RV outlier observations removed.
    :param L_HE: nested array, containing the data observed with the observation mode HE,
    with the low SNR and RV outlier observations removed.
    :param L_new_new: nested array, containing the data with the low SNR and RV outlier observations removed.
    '''
    
    #We only select the observations with favorable airmass
    L = L[airmass<airmass_limit]
    RV = RV[airmass<airmass_limit]
    SNR = SNR[airmass<airmass_limit]
    modes = modes[airmass<airmass_limit]
    
    #Array containing the indices of the flagged SNR values. These flagged indices will them be discarded from 
    #the array of interest, L.
    flag = []

    #Removing SNR outliers 
    #Define a measure of the spread. We use IQR since we are using the median as a measure of our data's central value.
    IQR = np.percentile(np.diff(SNR), 75)-np.percentile(np.diff(SNR), 25)

    #Looping over all the SNR values. We look at the difference between neighbouring points. In doing so, we hope to 
    #detect large fluctuations in the SNR.
    for i in range(len(np.diff(SNR))):
        #We flag indices only if their SNR difference is above a certain threshold, to avoid removing SNR values in a SNR chain.
        #A SNR chain is a sequence of points at very similar SNR values. During a night of observation, the SNR can jump from one
        #chain to another. What we want to remove is not these chains, but the random SNR outliers inside the chains.
        if np.abs(np.diff(SNR)[i]/np.median(np.diff(SNR))) > sigma_coeff_SNR*IQR:
            flag.append(i+1)
    
    #We remove the flagged indices from the SNR, mode, RV and L arrays.
    new_SNR = np.delete(SNR, flag[:-1], axis=0)
    new_L = np.delete(L, flag[:-1], axis=0)

    new_RV = np.delete(RV, flag[:-1], axis=0)
    new_modes = np.delete(modes, flag[:-1], axis=0)
    
    #Distinguish two cases depending on the number of modes of observation.
    #If there are two modes of observation.
    if np.sum(modes=='A') != len(modes) and np.sum(modes=='E')!= len(modes): 
        #We separate the spectra and their RV values based on their observing mode.
        L_HA = new_L[new_modes=='A']
        L_HE = new_L[new_modes=='E']

        RV_HA = new_RV[new_modes=='A']
        RV_HE = new_RV[new_modes=='E']

        SNR_HA = new_SNR[new_modes=='A']
        SNR_HE = new_SNR[new_modes=='E']
    
        #We remove the spectra with outlier RV values.
        L_HA = RV_clip(RV_HA, L_HA, sigma_coeff_RV)
        L_HE = RV_clip(RV_HE, L_HE, sigma_coeff_RV)

        return L_HA, L_HE
    #If there is one mode of observation.
    else:
        #We remove the spectra with outlier RV values.
        L_new_new = RV_clip(new_RV, new_L, sigma_coeff_RV)
        return L_new_new

    
def RV_clip(RV, L, sig):
    '''
    Function to remove the data with outlier values of RV. We do this by looking for RV values
    that are outside the RV data series by 20 sigma (IQR in our case since we are using median).
    Parameters
    ----------
    :param RV: array, containing the RV data along which we look for outliers.
    :param L: array, containing the data we are interested in and whose elements we will remove 
    if they are associated to outlier RV data points.
    :param sig:int, sigma clipping value. Number of sigma away from the average value that are accepted.
    Returns
    ----------
    :param L: array, containing the array L with the elements associated to outlier RV values removed.
    '''
    #Initializing an array to contain the indices of the outliers
    bad_indices = []
    
    #Defining the interquartile range
    IQR = np.percentile(RV, 75)-np.percentile(RV, 25)
    
    #Scanning the RV array for outliers
    for i in range(len(RV)):
        if RV[i] > (1+sig*IQR)*np.abs(np.median(RV)) or RV[i] < (1-sig*IQR)*np.abs(np.median(RV)):
                bad_indices.append(i)

    #Removing the outliers
    L = np.delete(L, bad_indices, axis=0)
    
    return L
        
        
def vac2air(wavelength):
    '''
    Function to transform the wavelength values in vacuum to air.
    Parameters
    ----------
    :param wavelength: float or array, containing the wavelength(s) in vacuum we want to convert.
    Returns
    ----------
    :param wavelength_air: float or array, containing the wavelength(s) in air.
    '''

# See VALD website here

# http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

# The formula comes from Birch and Downs (1994, Metrologia, 31, 315)

    s = 1e4 / wavelength # s = 10^4 / lamda_vac, lamda_vac is in Angstrom

    n_air = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)

    wavelength_air = wavelength / n_air
    return wavelength_air



def air2vac(wavelength):
    '''
    Function to transform the wavelength values in air to vacuum.
    Parameters
    ----------
    :param wavelength: float or array, containing the wavelength(s) in air we want to convert.
    Returns
    ----------
    :param wavelength_vacuum: float or array, containing the wavelength(s) in vacuum.
    '''

# See VALD website here

# http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

# The formula comes from Birch and Downs (1994, Metrologia, 31, 315)

    s = 1e4 / wavelength # s = 10^4 / lamda_air, lamda_air is in Angstrom

    n_vac = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)

    wavelength_vacuum = wavelength * n_vac
    return wavelength_vacuum

def bound(low, high, ref_L, L):
    '''
    Function to select a range of values in list L, bounded by the values in ref_L.
    For example selecting the flux values for a spectrum, over a range [low, high] of wavelengths.
    Parameters
    ----------
    :param low: float, for the low-end cutoff of the values in ref_L.
    :param high: float, for the high-end cutoff of the values in ref_L.
    :param ref_L: array, containing the reference values.
    :param L: array, that we want to 'trim' using the bounds on ref_L.
    Returns
    ----------
    :param temp3: array, containing the array L that has been bounded with the [low, high] range of values of ref_L.
    '''
    #Removing the low end of the reference array and the array of interest.
    temp1 = ref_L[ref_L > low]
    temp2 = L[ref_L > low]
    
    #Removing the high end of the array of interest. 
    temp3 = temp2[temp1 < high]
    
    return temp3

#Change the order of the parameters here and everywhere else in the other codes
def gaussian_L(x, A, offset, mu, std, lin):
    '''
    Function describing a Gaussian with a linear trend that we can fit to spectral lines.
    Parameters
    ----------
    :param x: array, values used to evaluate the linear-trend Gaussian.
    :param A: float, depth/amplitude of the Gaussian.
    :param offset: float, offset term of the linear trend.
    :param mu: float, position/expected value of the Gaussian.
    :param std: float, width/standard deviation of the Gaussian.
    :param lin: float, linear term of the linear trend.
    Returns
    ----------
    :param Gauss_l: array, containing the linear-trend Gaussian values.
    '''
    # Making the Gaussian distribution
    Gauss_l = (A * np.exp(-0.5*((x-mu)/std)**2)) + (lin*x+offset)
    return Gauss_l

def Voigt(x, offset, depth, center, sig, gam):
    '''
    Function describing a Voigt profile.
    Parameters
    ----------
    :param x: array, values used to evaluate the Voigt profile.
    :param offset: float, offset term.
    :param depth: float, depth/amplitude of the Voigt profile.
    :param center: float, central position of the Voigt profile.
    :param sig: float, standard deviation of the Normal distribution part of the Voigt profile.
    :param gam: float, half-width at half-maximum of the Cauchy distribution part of the Voigt profile.
    Returns
    ----------
    :param voigt: array, containing the Voigt profile values.
    '''
    #Making the Voigt profile
    voigt = offset-(depth*voigt_profile(x-center, sig, gam))
    return voigt

#############################
############HERE#############
#############################

def sinusoid(t, A, phase, offset, period):
    '''
    Function describing a basic sinusoidal.
    Parameters
    ----------
    :param x: array, values used to evaluate the sinusoidal model.
    :param A: float, amplitude of sinusoid.
    :param phase: float, phase offset of the sinusoid.
    :param offset: float, y-axis offset term.
    :param period: float, period of the sinusoidal model.
    Returns
    ----------
    :param sin_mod: array, containing the sinusoidal model values.
    '''
    sin_mod = offset + A*np.sin((2*np.pi*t/period) + phase)
    return sin_mod
    
def equivalent_width_calculator(wavelength, flux_vals, flux_errs, N, wav_ranges, low_lim, up_lim, plot):
    '''
    Routine to calculate the equivalent width of a line.
    Parameters
    ----------
    :param wavelength: array, wavelength range over which the line is present.
    :param flux_vals: array, flux values over the considered wavelength range.
    :param flux_errs: array, error on the flux values.
    :param N: int, number of points to interpolate on for a better determination.
    :param wav_ranges: list of tuples, containing the continuum chunks used for the equivalent width calculation.
    :param low_lim: float, lower limit of the wavelength range we want to calculate the equivalent width over.
    :param up_lim: float, upper limit of the wavelength range we want to calculate the equivalent width over.
    :param plot: bool, whether or not to plot the diagnostic plots.
    Returns
    ----------
    :param Eq_width: float, equivalent width of the line considered.
    '''
    #Interpolate line to have more points on the line
    ref_wavelength = np.linspace(low_lim, up_lim, N)
    
    inter_flux_vals = bound(low_lim, up_lim, wavelength, flux_vals)
    inter_flux_errs = bound(low_lim, up_lim, wavelength, flux_errs)
    interp_wavelength = bound(low_lim, up_lim, wavelength, wavelength)
    
    interpol_func_vals = interp1d(interp_wavelength, inter_flux_vals, kind='cubic', fill_value='extrapolate')
    interpol_func_errs = interp1d(interp_wavelength, inter_flux_errs**2, kind='cubic', fill_value='extrapolate')
    
    interp_flux_vals = interpol_func_vals(ref_wavelength)
    interp_flux_errs = np.sqrt(interpol_func_errs(ref_wavelength))

    ##Retrieving the spectrum, error and wavelengths in these ranges
    continuu_spctr = []
    continuu_wav = []
    continuu_err = []
    for x in wav_ranges:
        contin_spctr = bound(x[0], x[1], wavelength, flux_vals)
        contin_err = bound(x[0], x[1], wavelength, flux_errs)
        contin_wav = bound(x[0], x[1], wavelength, wavelength)
        continuu_spctr.append(contin_spctr)
        continuu_wav.append(contin_wav)
        continuu_err.append(contin_err)
    continuum_spctr = np.array(list(itertools.chain.from_iterable(continuu_spctr)))
    continuum_err = np.array(list(itertools.chain.from_iterable(continuu_err)))
    continuum_wav = np.array(list(itertools.chain.from_iterable(continuu_wav)))
    p_continuum = np.poly1d(np.polyfit(continuum_wav, continuum_spctr, 1, w=1/continuum_err**2))

    #Debugging
    if plot:
        #Plotting the continuum chunks
        plt.figure(figsize=[8, 5])
        plt.errorbar(wavelength, flux_vals, yerr=flux_errs, label='Data')
        plt.plot(wavelength, p_continuum(wavelength), color='orange', label='Polyn. model')
        for i, x in enumerate(wav_ranges):
            plt.axvline(x[0], color='r', linestyle='--')
            plt.axvline(x[1], color='r', linestyle='--')
            if i==0:
                plt.axvspan(x[0], x[1], color='r', alpha=0.3, label='Continuum Range')
            else:
                plt.axvspan(x[0], x[1], color='r', alpha=0.3)

        plt.axvline(10833.2, color='red', label='He triplet location')
        plt.axvline(10833.3, color='red')
        plt.axvline(10832.1, color='red')
        plt.axvline(10830.1, color='green')
        plt.axvline(10830.1, color='green', label='Si line location')
        plt.ylabel('Normalized Flux')
        plt.xlabel('Wavelength ($\AA$)')
        plt.xlim(wav_ranges[0][0]-5, wav_ranges[-1][1]+5)
        plt.legend()
        plt.show()
        
        #Plotting the equivalent width calculation
        plt.figure(figsize=[8, 5])
        plt.errorbar(wavelength, flux_vals, yerr=flux_errs, fmt='b--', label='Initial')
        plt.errorbar(ref_wavelength, interp_flux_vals, yerr=interp_flux_errs, fmt='r.', label='Interpolated')
        plt.plot(wavelength, p_continuum(wavelength), 'k--', label='Continuum level')
        plt.xlabel('Wavelength ($\AA$)')
        plt.ylabel('Normalized Flux')
        plt.xlim(low_lim, up_lim)
        plt.ylim(0.9, 1.01)
        plt.title('Equivalent Width plot')
        plt.legend()
        plt.show()

    #Approximating the equivalent width as the sum
    Eq_width = 0.0
    Eq_width_err = 0.0
    
    if len(flux_errs)!=0:
        for i in range(1, len(ref_wavelength)):        
            Eq_width += (ref_wavelength[i]-ref_wavelength[i-1]) * (p_continuum(ref_wavelength)[i]-interp_flux_vals[i])
            Eq_width_err += (ref_wavelength[i]-ref_wavelength[i-1])**2 * interp_flux_errs[i]**2

        Eq_width_err = np.sqrt(Eq_width_err)
    
        return Eq_width, Eq_width_err
    else:
        for i in range(1, len(ref_wavelength)):        
            Eq_width += (ref_wavelength[i]-ref_wavelength[i-1]) * (p_continuum(ref_wavelength)[i]-interp_flux_vals[i])
            
        return Eq_width


def fit_spctr_line(fit_func, low_lim, up_lim, low_lim_ew, up_lim_ew, ini_guess, guess_bounds, x, y, y_err, c, wav_ranges, param_names, plot=True, N=100):
    '''
    Routine to fit a function, fit_func, to a spectral line located in the wavelength range [low_lim, up_lim].
    We give as input a guess for the best-fit parameters and the acceptable bounds within which the algorithm
    should search for the best-fit.
    Parameters
    ----------
    :param fit_func: function, that we want to fit the spectral line with.
    :param low_lim: float, lower limit of the wavelength range we want to fit over.
    :param upper_lim: float, upper limit of the wavelength range we want to fit over.
    :param low_lim_ew: float, lower limit of the wavelength range we want to calculate the equivalent width over.
    :param upper_lim_ew: float, upper limit of the wavelength range we want to calculate the equivalent width over.
    :param ini_guess: array, containing an initial guess/starting point for the best-fit parameter values.
    :param guess_bounds: array, containing the bounds, with respect to the initial guess, 
    within which we search for the best-fit parameters.
    :param x: array, containing the wavelength values.
    :param y: array, containing the data we want to fit(e.g. flux values of spectra).
    :param y_err: array, containing the error values on the data we want to fit.
    :param c: string, the color of the best-fit model when plotting.
    :param wav_ranges: list of tuples, containing the continuum chunks used for the equivalent width calculation.
    :param param_names: list, containing the name of the parameters in fit_func we want to fit for.
    :param plot: bool, whether or not to plot the diagnostic plots.
    :param N: int, number of points to interpolate on for the equivalent width calculation.
    
    Returns
    ----------
    :param thetas: array, containing best-fit parameters for the fit to each (x[i], y[i]) pair.
    :param err: array, containing the error bars on the best-fit parameters for the fit to each (x[i], y[i]) pair.
    :param lmfit_thetas: array, containing best-fit parameters for the fit to each (x[i], y[i]) pair, using lmfit.
    :param lmfit_err: array, containing the error bars on the best-fit parameters for the fit to each (x[i], y[i]) pair, 
    using lmfit.
    '''

    #Initializing the arrays containing the best-fit parameters and the errors on them.
    thetas = np.ones((len(y), len(ini_guess)+1))
    lmfit_thetas = np.ones((len(y), len(ini_guess)+1))
    lmfit_err = np.ones((len(y), len(ini_guess)+1))
    err = np.ones((len(y), len(ini_guess)+1))
    
    #Looping over all the arrays(/spectra).
    for i in range(len(x)):
        
        #Creating the bound versions of the x[i], y[i] and y_err[i] arrays. Bound over the 
        # wavelength range of interest.
        bound_x = bound(low_lim, up_lim, x[i], x[i])
        bound_y = bound(low_lim, up_lim, x[i], y[i])
        if len(y_err)!=0:
            bound_y_err = bound(low_lim, up_lim, x[i], y_err[i])
        
        #Using the curve_fit function from scipy to fit the function of interest to the data.
        if len(y_err)!=0:
            params, cov = curve_fit(fit_func, bound_x, bound_y, sigma=bound_y_err, p0 = ini_guess, bounds = guess_bounds)
        else:
            params, cov = curve_fit(fit_func, bound_x, bound_y, p0 = ini_guess, bounds = guess_bounds)
        #Extracting the best-fit parameters and the error on the best-fit parameters.
        thetas[i][:-1] = params
        err[i][:-1] = np.sqrt(np.diag(cov))

        #Using lmfit
        lm_fit_func = lf.Model(fit_func)
        param = lf.Parameters()
        for j in range(len(ini_guess)):
            param.add(param_names[j], value=ini_guess[j], min=guess_bounds[0][j], max=guess_bounds[1][j])

        result = lm_fit_func.fit(bound_y, x=bound_x, params=param, weights=1/bound_y_err**2)
        print(result.fit_report())
        
        for h in range(len(ini_guess)):
            lmfit_thetas[i][h] = result.params[param_names[h]].value
            lmfit_err[i][h] = result.params[param_names[h]].stderr

        #Getting the equivalent width 
        #spectrum_obj = Spectrum1D(flux = bound_y*u.Jy, spectral_axis = bound_x*u.AA)
        #eq_width2 = equivalent_width(spectrum_obj)
        if len(y_err)!=0:
            eq_width, eq_width_err = equivalent_width_calculator(x[i], y[i], y_err[i], N, wav_ranges, low_lim_ew, up_lim_ew, plot) 
            thetas[i][-1] = eq_width
            err[i][-1] = eq_width_err
            lmfit_err[i][-1] = eq_width_err
        
        else:
            eq_width = equivalent_width_calculator(x[i], y[i], [], N, wav_ranges, low_lim_ew, up_lim_ew, plot) 
            thetas[i][-1] = eq_width
        
        lmfit_thetas[i][-1] = eq_width
    
        #Creating the best-fit model for plotting purposes.
        model_x = np.linspace(low_lim, up_lim, N)
        model_curve_fit = fit_func(model_x, *params)
        model_lmfit = lm_fit_func.eval(params = result.params, x=model_x)
 
        #Plotting the best-fit model on top of the data.
        if plot:
            plt.figure(figsize=[8, 5])
            if len(y_err)!=0:
                plt.errorbar(bound_x, bound_y, yerr=bound_y_err,  fmt='b.', label='data')
            else:
                plt.plot(bound_x, bound_y, 'b.', label='data')
            plt.plot(model_x, model_curve_fit, c, label='Curve fit')
            plt.plot(model_x, model_lmfit, 'g', label='Lmfit')
            plt.xlabel('Wavelength ($\AA$)')
            plt.ylabel('Normalized Flux')
            plt.legend()
            for j in range(len(thetas[i])):
                print(thetas[i][j], err[i][j])
            plt.show()

    return thetas, err, lmfit_thetas, lmfit_err

def fit_spctr_line_special(fit_func, include_ranges, low_lim, up_lim, low_lim_ew, up_lim_ew, ini_guess, guess_bounds, x, y, y_err, c, wav_ranges, param_names, plot=True, N=100):
    '''
    Routine to fit a function, fit_func, to a spectral line located in the wavelength range [low_lim, up_lim].
    We give as input a guess for the best-fit parameters and the acceptable bounds within which the algorithm
    should search for the best-fit.
    Parameters
    ----------
    :param fit_func: function, that we want to fit the spectral line with.
    :param include_ranges: list of tuples, containing the wavelength ranges to include in our analysis.
    :param low_lim: float, lower limit of the wavelength range we want to fit over.
    :param upper_lim: float, upper limit of the wavelength range we want to fit over.
    :param low_lim_ew: float, lower limit of the wavelength range we want to calculate the equivalent width over.
    :param upper_lim_ew: float, upper limit of the wavelength range we want to calculate the equivalent width over.
    :param ini_guess: array, containing an initial guess/starting point for the best-fit parameter values.
    :param guess_bounds: array, containing the bounds, with respect to the initial guess, 
    within which we search for the best-fit parameters.
    :param x: array, containing the wavelength values.
    :param y: array, containing the data we want to fit(e.g. flux values of spectra).
    :param y_err: array, containing the error values on the data we want to fit.
    :param c: string, the color of the best-fit model when plotting.
    :param wav_ranges: list of tuples, containing the continuum chunks used for the equivalent width calculation.
    :param param_names: list, containing the name of the parameters in fit_func we want to fit for.
    :param plot: bool, whether or not to plot the diagnostic plots.
    :param N: int, number of points to interpolate on for the equivalent width calculation.
    
    Returns
    ----------
    :param thetas: array, containing best-fit parameters for the fit to each (x[i], y[i]) pair.
    :param err: array, containing the error bars on the best-fit parameters for the fit to each (x[i], y[i]) pair.
    :param lmfit_thetas: array, containing best-fit parameters for the fit to each (x[i], y[i]) pair, using lmfit.
    :param lmfit_err: array, containing the error bars on the best-fit parameters for the fit to each (x[i], y[i]) pair, 
    using lmfit.
    '''

    #Initializing the arrays containing the best-fit parameters and the errors on them.
    thetas = np.ones((len(y), len(ini_guess)+1))
    lmfit_thetas = np.ones((len(y), len(ini_guess)+1))
    lmfit_err = np.ones((len(y), len(ini_guess)+1))
    err = np.ones((len(y), len(ini_guess)+1))
    
    #Looping over all the arrays(/spectra).
    for i in range(len(x)):
        
        #Creating the bound versions of the x[i], y[i] and y_err[i] arrays. Bound over the 
        # wavelength range of interest.
        boun_x = bound(low_lim, up_lim, x[i], x[i])
        boun_y = bound(low_lim, up_lim, x[i], y[i])
        if len(y_err)!=0:
            boun_y_err = bound(low_lim, up_lim, x[i], y_err[i])
        
        #Only keep the wavelength ranges we want to consider
        includ_lamda = []
        includ_flux = []
        includ_err = []
        for m in include_ranges:
            includ_lamda.append(bound(m[0], m[1], boun_x, boun_x))
            includ_flux.append(bound(m[0], m[1], boun_x, boun_y))
            if len(y_err)!=0:
                includ_err.append(bound(m[0], m[1], boun_x, boun_y_err))

        bound_x = np.array(list(itertools.chain.from_iterable(includ_lamda)))
        bound_y = np.array(list(itertools.chain.from_iterable(includ_flux)))
        if len(y_err)!=0:
            bound_y_err = np.array(list(itertools.chain.from_iterable(includ_err)))
        
        if plot and i==0:
            if len(y_err)!=0:
                plt.errorbar(boun_x, boun_y, yerr=boun_y_err, fmt='b.', label='Excluded')
                plt.errorbar(bound_x, bound_y, yerr=bound_y_err, fmt='r.', label='Included')
            else:
                plt.plot(boun_x, boun_y,'b.', label='Excluded')
                plt.plot(bound_x, bound_y, 'r.', label='Included')
            plt.legend()
            plt.show()
        
        #Using the curve_fit function from scipy to fit the function of interest to the data.
        if len(y_err)!=0:
            params, cov = curve_fit(fit_func, bound_x, bound_y, sigma=bound_y_err, p0 = ini_guess, bounds = guess_bounds)
        else:
            params, cov = curve_fit(fit_func, bound_x, bound_y, p0 = ini_guess, bounds = guess_bounds)
        #Extracting the best-fit parameters and the error on the best-fit parameters.
        thetas[i][:-1] = params
        err[i][:-1] = np.sqrt(np.diag(cov))

        #Using lmfit
        lm_fit_func = lf.Model(fit_func)
        param = lf.Parameters()
        for j in range(len(ini_guess)):
            param.add(param_names[j], value=ini_guess[j], min=guess_bounds[0][j], max=guess_bounds[1][j])
        if len(y_err)!=0:
            result = lm_fit_func.fit(bound_y, x=bound_x, params=param, weights=1/bound_y_err**2)
        else:
            result = lm_fit_func.fit(bound_y, x=bound_x, params=param)
        print(result.fit_report())
        
        for h in range(len(ini_guess)):
            lmfit_thetas[i][h] = result.params[param_names[h]].value
            lmfit_err[i][h] = result.params[param_names[h]].stderr

        #Getting the equivalent width 
        #spectrum_obj = Spectrum1D(flux = bound_y*u.Jy, spectral_axis = bound_x*u.AA)
        #eq_width2 = equivalent_width(spectrum_obj)
        if len(y_err)!=0:
            eq_width, eq_width_err = equivalent_width_calculator(x[i], y[i], y_err[i], N, wav_ranges, low_lim_ew, up_lim_ew, plot) 
            thetas[i][-1] = eq_width
            err[i][-1] = eq_width_err
            lmfit_err[i][-1] = eq_width_err
        
        else:
            eq_width = equivalent_width_calculator(x[i], y[i], [], N, wav_ranges, low_lim_ew, up_lim_ew, plot) 
            thetas[i][-1] = eq_width
        
        lmfit_thetas[i][-1] = eq_width
    
        #Creating the best-fit model for plotting purposes.
        model_x = np.linspace(low_lim, up_lim, N)
        model_curve_fit = fit_func(model_x, *params)
        model_lmfit = lm_fit_func.eval(params = result.params, x=model_x)
        #Plotting the best-fit model on top of the data.
        if plot:
            plt.figure(figsize=[8, 5])
            if len(y_err)!=0:
                plt.errorbar(bound_x, bound_y, yerr=bound_y_err,  fmt='b.', label='data')
            else:
                plt.plot(bound_x, bound_y, 'b.', label='data')
            plt.plot(model_x, model_curve_fit, c, label='Curve fit')
            plt.plot(model_x, model_lmfit, 'g', label='Lmfit')
            plt.xlabel('Wavelength ($\AA$)')
            plt.ylabel('Normalized Flux')
            plt.legend()
            for j in range(len(thetas[i])):
                print(thetas[i][j], err[i][j])
            plt.show()

    return thetas, err, lmfit_thetas, lmfit_err


def plot_TS_Periodo(T, L, L_err, title1, title2, mode, save=True, fit=False, order=1, N=1000):
    '''
    Function to plot the Time series and periodogram of a particular quantity of interest L.
    There is also the option of fitting a polynomial to the time series of the quantity of interest.
    Parameters
    ----------
    :param T: array, timestamp for the data contained in L.
    :param L: array, data for the quantity of interest.
    :param title1: string, used to label the y-axis of the time series plot.
    :param title2: string, used as the title for the time series and periodogram.
    :param mode: string, if there are multiple modes of observation, it is used in the title 
    to say for which mode of observation we are plotting the quantity of interest.
    :param save: bool, whether or not to save the Time series and periodogram to a PDF.
    :param fit: bool, whether or not to fit a polynomial to the time series.
    :param order: int, if fit=True, the order of the polynomial to fit to the time series.
    :param N: int, number of points to plot the polynomial with.
    Returns
    ----------
     '''
    oscillation_freq = 24*60/5.4
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[15, 4])
    
    if len(L_err)!=0:
        ax1.errorbar(T, L, yerr=L_err, fmt='.', label='Data')
    else:
        ax1.plot(T, L, '.', label='Data')
        
    ax1.set_xlabel('Time (MJD)')
    ax1.set_ylabel(title1)
    ax1.set_title(title2+' Time Series '+mode)
    if fit:
        if len(L_err)!=0:
            model_poly = np.poly1d(np.polyfit(T, L, order, w=1/L_err**2))
        else:
            model_poly = np.poly1d(np.polyfit(T, L, order))
        model_x = np.linspace(min(T), max(T), N)
        print(model_poly)
        ax1.plot(model_x, model_poly(model_x), 'r', label=str(order)+'th order polynomial fit')
        ax1.legend()

    Periodo = LombScargle(T, L).autopower(nyquist_factor=1)
    ax2.loglog(Periodo[0], Periodo[1])
    ax2.set_xlabel('Frequency (1/d)')
    ax2.set_ylabel('Power')
    ax2.set_title(title2+' Periodogram '+mode)
    
    ax2.axvline(oscillation_freq, color='r', label='Oscillation Frequency')
    ax2.legend()
    if save:
        plt.savefig('Plot_Results/TS_Periodo_'+title2+'_'+mode+'.pdf')
    plt.show()

    
def SNR_calculator(low, high, tot_lamda, tot_spctr, tot_err):
    '''
    Function to calculate the SNR manually using a part of the continuum of the spectrum
    that does not present any important variations due to absorption/emission lines.
    Parameters
    ----------
    :param low: float, lower wavelength limit for the continuum.
    :param high: float, upper wavelength limit for the continuum.
    :param tot_lamda: array, containing the wavelength values of the spectrum.
    :param tot_spctr: array, containing the flux values of the spectrum. 
    :param tot_err: array, containing the flux error values of the spectrum.
    Returns
    :param measured_SNR: array, containing the calculated SNR values for all the spectrum.
    ----------
    '''
    #Initiating the SNR array.
    measured_SNR = np.ones(tot_lamda.shape[0])
    
    #Populating the SNR array.
    for h in range(len(tot_lamda)):
        
        #Defining the continuum
        continuum_lam = bound(low, high, tot_lamda[h], tot_lamda[h])
        continuum_spctr = bound(low, high, tot_lamda[h], tot_spctr[h])
        continuum_err = bound(low, high, tot_lamda[h], tot_err[h])

        #Getting the weighted average and standard deviation of the flux in the continuum
        avg = np.average(continuum_spctr, weights = 1/continuum_err**2)
        variance = np.average((continuum_spctr - avg)**2, weights = 1/continuum_err**2)
        std = np.sqrt(variance)
        
        #Calculating the SNR from the average and standard deviation of the continuum flux
        measured_SNR[h] = avg/std
        
    return measured_SNR

def Correlation_Plot(mode, A, B, A_err, B_err, titleA, titleB, title, day, save=True):
    '''
    Function to create a correlation plot between two arrays A and B.
    Parameters
    ----------
    :param mode: array, used to divide the data based on the mode of observation used.
    :param A: array, values to plot on x-axis.
    :param B: array, values to plot on y-axis.
    :param A_err: array, containing error on the values contained in A.
    :param B_err: array, containing error on the values contained in B. 
    :param titleA: string, title for the x-axis.
    :param titleB: string, title for the y-axis.
    :param title: string, title for the plot.
    Returns
    :param measured_SNR: array, containing the calculated SNR values for all the spectrum.
    ----------
    '''
    if np.sum(mode=='A') != len(mode) and np.sum(mode=='E')!= len(mode):
        
        A_HA = A[mode=='A']
        A_HE = A[mode=='E']
        
        B_HA = B[mode=='A']
        B_HE = B[mode=='E']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[15, 4], sharey=True)

        if len(A_err)!=0 and len(B_err)!=0:
            A_err_HA = A_err[mode=='A']
            A_err_HE = A_err[mode=='E']
            B_err_HA = B_err[mode=='A']
            B_err_HE = B_err[mode=='E']
            ax1.errorbar(A_HA, B_HA, xerr=A_err_HA, yerr=B_err_HA, fmt='b.')
            ax2.errorbar(A_HE, B_HE, xerr=A_err_HE, yerr=B_err_HE, fmt='g.')
            
        elif len(A_err)==0 and len(B_err)!=0:
            B_err_HA = B_err[mode=='A']
            B_err_HE = B_err[mode=='E']
            ax1.errorbar(A_HA, B_HA, yerr=B_err_HA, fmt='b.')
            ax2.errorbar(A_HE, B_HE, yerr=B_err_HE, fmt='g.')
        elif len(A_err)!=0 and len(B_err)==0:
            A_err_HA = A_err[mode=='A']
            A_err_HE = A_err[mode=='E']
            ax1.errorbar(A_HA, B_HA, xerr=A_err_HA, fmt='b.')
            ax2.errorbar(A_HE, B_HE, xerr=A_err_HE, fmt='g.')
        else:
            ax1.plot(A_HA, B_HA, 'b.')
            ax2.plot(A_HE, B_HE, 'g.')

        ax1.set_xlabel(titleA)
        ax2.set_xlabel(titleA)
        ax1.set_ylabel(titleB)
        ax1.set_title(title+' correlation for '+day+' (HA)')
        ax2.set_title(title+' correlation for '+day+' (HE)')

        textstr_HA = '\n'.join((r"$r_P = %.3f$" % (np.corrcoef(A_HA, B_HA)[0][1], ), 
                            r"$r_S = %.3f$" % (ss.spearmanr(A_HA, B_HA).correlation, )))
        ax1.text(0.79, 0.80, textstr_HA, transform=ax1.transAxes, fontsize=12, bbox = dict(facecolor='white', alpha=0.5))
        
        textstr_HE = '\n'.join((r"$r_P = %.3f$" % (np.corrcoef(A_HE, B_HE)[0][1], ), 
                            r"$r_S = %.3f$" % (ss.spearmanr(A_HE, B_HE).correlation, )))
        ax2.text(0.79, 0.80, textstr_HE, transform=ax2.transAxes, fontsize=12, bbox = dict(facecolor='white', alpha=0.5))
        
        if save:
            plt.savefig('/Users/samsonmercier/Desktop/Correlation-'+day[-2:]+'/'+title+'-'+day[-2:]+'.pdf')

    else:
        fig, ax = plt.subplots(1, 1, figsize=[7, 4])

        if len(A_err)!=0 and len(B_err)!=0:
            ax.errorbar(A, B, xerr=A_err, yerr=B_err, fmt='.')
        elif len(A_err)==0 and len(B_err)!=0:
            ax.errorbar(A, B, yerr=B_err, fmt='.')
        elif len(A_err)!=0 and len(B_err)==0:
            ax.errorbar(A, B, xerr=A_err, fmt='.')
        else:
            ax.plot(A, B, '.')

        ax.set_xlabel(titleA)
        ax.set_ylabel(titleB)
        ax.set_title(title+' correlation for '+day)
        textstr = '\n'.join((r"$r_P = %.3f$" % (np.corrcoef(A, B)[0][1], ), 
                            r"$r_S = %.3f$" % (ss.spearmanr(A, B).correlation, )))
        ax.text(0.79, 0.80, textstr, transform=ax.transAxes, fontsize=12, bbox = dict(facecolor='white', alpha=0.5))
        if save:
            plt.savefig('/Users/samsonmercier/Desktop/'+title+'-'+day[-2:]+'.pdf')


def new_extraction(file_directory, blaze_directory, CCF_directory, telluric_directory, rassine_directory, order, wav_ranges, Rassine=False, plot=False):
    '''
    Function to extract the important quantities from the FITS files for a given day of solar observations.
    Parameters
    ----------
    :param file_directory: string, name of directory containing the spectral FITS files for a given day of solar observations.
    :param blaze_directory: string, name of directory containing the blaze FITS files for a given day of solar observations.
    Generally, there will be one or two files depending on the number of modes of observation.
    :param CCF_directory: string, name of directory containing the CCF FITS files for a given day of solar observations.
    These files contain information about the RV of each spectrum.
    :param telluric_directory: string, name of directory containing the telluric FITS files for a given day of solar observations.
    :param rassine_directory: string, name of directory containing the RASSINE normalized spectra. 
    These files contain information about the telluric model used for the telluric correction.
    :param order: int, order of the Échelle spectrograph we want to use.
    :param wav_ranges: list of tuples, containing the continuum chunks used for the normalization.
    :param plot: bool, whether or not to plot the normalization diagnostic plot.
    Returns
    ----------
    :param total_lamda: nested array, containing list of wavelengths for each spectrum, for the given order.
    :param total_spctr: nested array, containing list of flux values for each spectrum, for the given order.
    :param total_norm_spctr: nested array, containing list of normalized flux values for each spectrum, for the given order.
    :param total_err: nested array, containing list of errors on the flux values for each spectrum, for the given order.
    :param total_norm_err: nested array, containing list of errors on the normalized flux values for each spectrum, 
    for the given order.
    :param total_SNR: nested array, containing the SNR value for each spectrum, for the given order.
    :param mode: nested array, containing the mode of observation for each spectrum.
    :param date: nested array, containing timestamp of acquisition for each spectrum.
    :param total_RV: nested array, containing the RV values for each spectrum, obtained with a CCF.
    :param total_RV_err: nested array, containing the error on the RV values for each spectrum, obtained with a CCF.
    :param total_FWHM: nested array, containing the FWHM values for the CCF of each spectrum.
    :param total_FWHM_err: nested array, containing the error on the FWHM for the CCF of each spectrum.
    :param total_BIS_SPAN: nested array, containing the Bisector Span values for the CCF of each spectrum.
    :param total_BIS_SPAN_err: nested array, containing the error on the Bisector Span for the CCF of each spectrum.
    :param total_CONTRAST: nested array, containing the Contrast values for the CCF of each spectrum.
    :param total_CONTRAST_err: nested array, containing the error on the Contrast for the CCF of each spectrum.
    :param total_H2O: nested array, containing the integrated column density of H2O for each spectrum.
    :param total_H2O_err: nested array, containing the error on the integrated column density of H2O for each spectrum.
    :param total_O2: nested array, containing the integrated column density of O2 for each spectrum.
    :param total_O2_err: nested array, containing the error on the integrated column density of O2 for each spectrum.    
    :param total_CO2: nested array, containing the integrated column density of CO2 for each spectrum.
    :param total_CO2_err: nested array, containing the error on the integrated column density of CO2 for each spectrum.    
    :param total_AIRM: nested array, containing the airmass at the time of observation for each spectrum.
    '''
    print('INITIALIZATION')
    #Initialize the arrays.
    #Contains the wavelength for the order of interest for the spectra.
    total_lamda = np.zeros((len(os.listdir(file_directory)), 4084))
    
    #Contains the flux and normalized values of the spectra of the order of interest.
    total_spctr = np.zeros((len(os.listdir(file_directory)), 4084))
    total_norm_spctr = np.zeros((len(os.listdir(file_directory)), 4084))
    
    #Contains the flux of the telluric model of the order of interest.
    total_telluric = np.zeros((len(os.listdir(file_directory)), 4084))

    #Contains the error on the flux and normalized flux values of the spectra.
    total_err = np.zeros((len(os.listdir(file_directory)), 4084))
    total_norm_err = np.zeros((len(os.listdir(file_directory)), 4084))
    
    #Contains the SNR value of the spectra.
    total_SNR = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the mode of each spectrum.
    mode = np.zeros((len(os.listdir(file_directory))), dtype=str)
    
    #Contains the timestamp at which each spectrum was collected (?).
    date = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the RV values of the spectra.
    #The RV values are obtained with a CCF routine, that has been TC.
    total_RV = np.zeros((len(os.listdir(file_directory))))
    
    #Error on the RV values of the spectra.
    total_RV_err = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the FWHM of the CCF of the spectra.
    total_FWHM = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the error on the FWHM of the CCF of the spectra.
    total_FWHM_err = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the Bisector span of the CCF of the spectra.
    total_BIS_SPAN = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the error on the Bisector span of the CCF of the spectra.
    total_BIS_SPAN_err = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the Contrast of the CCF of the spectra.
    total_CONTRAST = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the error on the Contrast of the CCF of the spectra.
    total_CONTRAST_err = np.zeros((len(os.listdir(file_directory))))

    #Contains the integrated column density of H2O, CO2 and O2 at the time of acquisition of the spectra.
    total_H2O = np.zeros((len(os.listdir(file_directory))))
    total_O2 = np.zeros((len(os.listdir(file_directory))))
    total_CO2 = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the error on the integrated column density of H2O, CO2 and O2 at the time of acquisition of the spectra.
    total_H2O_err = np.zeros((len(os.listdir(file_directory))))
    total_O2_err = np.zeros((len(os.listdir(file_directory))))
    total_CO2_err = np.zeros((len(os.listdir(file_directory))))

    #Contains the airmass at the time of observation of each spectrum.
    total_AIRM = np.zeros((len(os.listdir(file_directory))))
    
    #Defining the Sun's BB for later.
    Sun_BB = models.BlackBody(temperature = 5778*u.K)

    #Initial mode analysis to see the modes used in the data.
    for i in range(len(os.listdir(file_directory))):
        file = pf.open(file_directory+'/'+sorted(os.listdir(file_directory))[i])
        mode[i] = file[0].header['HIERARCH ESO INS MODE'][1]
    
    print('EXTRACTION + NORMALIZATION')
    #Loop over all the files in the directory and populate the arrays.
    for i in range(len(os.listdir(file_directory))):
        #Opening the files in the directories.
        file = pf.open(file_directory+'/'+sorted(os.listdir(file_directory))[i])
        file_CCF = pf.open(CCF_directory +'/'+sorted(os.listdir(CCF_directory))[i])
        file_telluric = pf.open(telluric_directory +'/'+sorted(os.listdir(telluric_directory))[i])
        
        #Extracting the telluric model 
        total_telluric[i] = file_telluric[6].data[order]
        
        #Extracting the wavelength.
        total_lamda[i] = file[4].data[order]

        #Extracting the DLL for spectrum correction - only used later.
        file_DLL = file[6].data[order]
        
        #Extracting the timestamp of each spectra.
        date[i] = file[0].header['MJD-OBS']

        #Getting the RV and error on the RV from the CCF files.
        total_RV[i] = file_CCF[0].header['HIERARCH ESO QC CCF RV']
        total_RV_err[i] = file_CCF[0].header['HIERARCH ESO QC CCF RV ERROR']
        
        #Getting the error and value on the FWHM.
        total_FWHM[i] = file_CCF[0].header['HIERARCH ESO QC CCF FWHM']
        total_FWHM_err[i] = file_CCF[0].header['HIERARCH ESO QC CCF FWHM ERROR']
        
        #Getting the error and value on the Bisector Span.
        total_BIS_SPAN[i] = file_CCF[0].header['HIERARCH ESO QC CCF BIS SPAN']
        total_BIS_SPAN_err[i] = file_CCF[0].header['HIERARCH ESO QC CCF BIS SPAN ERROR']

        #Getting the error and value on the Contrast.
        total_CONTRAST[i] = file_CCF[0].header['HIERARCH ESO QC CCF CONTRAST']
        total_CONTRAST_err[i] = file_CCF[0].header['HIERARCH ESO QC CCF CONTRAST ERROR']
        
        #Getting the airmass .
        total_AIRM[i] = (file[0].header['HIERARCH ESO TEL AIRM START'] + file[0].header['HIERARCH ESO TEL AIRM END'])/2
    
        #Getting the error and value of the integrated column density for H2O, O2, CO2.
        total_H2O[i] = file[0].header['HIERARCH ESO QC TELL H2O IWV']
        total_H2O_err[i] = file[0].header['HIERARCH ESO QC TELL H2O IWV ERR']
        total_O2[i] = file[0].header['HIERARCH ESO QC TELL O2 IWV']
        total_O2_err[i] = file[0].header['HIERARCH ESO QC TELL O2 IWV ERR']
        total_CO2[i] = file[0].header['HIERARCH ESO QC TELL CO2 IWV']
        total_CO2_err[i] = file[0].header['HIERARCH ESO QC TELL CO2 IWV ERR']
    
        #Retrieving the SNR of each spectra at the order of interest.
        total_SNR[i] = file[0].header['HIERARCH ESO QC ORDER'+str(order)+' SNR']
        
        #Getting the raw spectrum, that has bee Telluric-Corrected(TC).
        file_spctr = file[1].data[order]
        file_err = file[2].data[order]

        #Distinguish two cases depending on the number of modes of observation.
        #If there are two modes of observation.
        if np.sum(mode=='A') != len(mode) and np.sum(mode=='E')!= len(mode):
            #Looking through the Blaze directory to find the Blaze files for each observation mode.
            for j in range(len(os.listdir(blaze_directory))):
                file_blaze = pf.open(blaze_directory+'/'+os.listdir(blaze_directory)[j])
                if file_blaze[0].header['HIERARCH ESO INS MODE'][1] == 'A':
                    blaze_HA = file_blaze
                else:
                    blaze_HE = file_blaze
                         
            #Getting the Blaze spectra for each mode of observation.
            blaze_HA_spctr = blaze_HA[1].data[order]
            blaze_HE_spctr = blaze_HE[1].data[order]
            
            #Performing the Blaze correction -- the Blaze file used depends on the observation mode used.
            if mode[i] == 'A':
                BC_spctr = file_spctr/blaze_HA_spctr
                BC_err = file_err/blaze_HA_spctr
            else:
                BC_spctr = file_spctr/blaze_HE_spctr
                BC_err = file_err/blaze_HE_spctr

            #Removing the trend of the spectrum's continuum using DLL and the Black Body of the Sun.
            BC_DLL_spctr = BC_spctr/file_DLL
            BC_DLL_err = BC_err/file_DLL
            
            total_spctr[i] = BC_DLL_spctr/Sun_BB(total_lamda[i]*u.AA).value
            total_err[i] = BC_DLL_err/Sun_BB(total_lamda[i]*u.AA).value
                                
        #If there is only one mode of observation.
        else:
            #Getting the blaze file.
            blaze = pf.open(blaze_directory+'/'+os.listdir(blaze_directory)[0])
            
            #Getting the blaze spectra.
            blaze_spctr = blaze[1].data[order]
            
            #Performing the Blaze correction.
            BC_spctr = file_spctr/blaze_spctr
            BC_err = file_err/blaze_spctr
            
            #Removing the trend of the spectrum's continuum using DLL and the Sun's Black Body spectrum.
            BC_DLL_spctr = BC_spctr/file_DLL
            BC_DLL_err = BC_err/file_DLL
            
            total_spctr[i] = BC_DLL_spctr/Sun_BB(file[4].data[order]*u.AA).value
            total_err[i] = BC_DLL_err/Sun_BB(file[4].data[order]*u.AA).value
            
        ##Retrieving the spectrum, error and wavelengths in these ranges
        continuu_spctr = []
        continuu_wav = []
        continuu_err = []
        for x in wav_ranges:
            contin_spctr = bound(x[0], x[1], total_lamda[i], total_spctr[i])
            contin_err = bound(x[0], x[1], total_lamda[i], total_err[i])
            contin_wav = bound(x[0], x[1], total_lamda[i], total_lamda[i])
            continuu_spctr.append(contin_spctr)
            continuu_wav.append(contin_wav)
            continuu_err.append(contin_err)
        continuum_spctr = np.array(list(itertools.chain.from_iterable(continuu_spctr)))
        continuum_err = np.array(list(itertools.chain.from_iterable(continuu_err)))
        continuum_wav = np.array(list(itertools.chain.from_iterable(continuu_wav)))

        ##Fitting the continuum range with  a 1st order polynomial 
        p_continuum = np.poly1d(np.polyfit(continuum_wav, continuum_spctr, 1, w=1/continuum_err**2))

        ##Making the normalized spectra and error bars.
        total_norm_spctr[i] = total_spctr[i]/p_continuum(total_lamda[i])
        total_norm_err[i] = total_err[i]/p_continuum(total_lamda[i])

        ##Plotting
        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[14, 8], sharex=True)
            ax1.errorbar(total_lamda[i], total_spctr[i], yerr=total_err[i], fmt='b', label='Data')
            ax1.plot(total_lamda[i], p_continuum(total_lamda[i]), color='orange', label='Polyn. model')
            ax2.errorbar(total_lamda[i], total_norm_spctr[i], yerr=total_norm_err[i], fmt='b')
            for i, x in enumerate(wav_ranges):
                ax1.axvline(x[0], color='r', linestyle='--')
                ax2.axvline(x[0], color='r', linestyle='--')
                ax1.axvline(x[1], color='r', linestyle='--')
                ax2.axvline(x[1], color='r', linestyle='--')
                if i==0:
                    ax1.axvspan(x[0], x[1], color='r', alpha=0.3, label='Continuum Range')
                else:
                    ax1.axvspan(x[0], x[1], color='r', alpha=0.3)
                ax2.axvspan(x[0], x[1], color='r', alpha=0.3)

            ax1.axvline(10833.2, color='red', label='He triplet location')
            ax1.axvline(10833.3, color='red')
            ax1.axvline(10832.1, color='red')
            ax1.axvline(10830.1, color='green')
            ax1.axvline(10830.1, color='green', label='Si line location')
            ax2.axvline(10833.2, color='red')
            ax2.axvline(10833.3, color='red')
            ax2.axvline(10832.1, color='red')
            ax2.axvline(10830.1, color='green')
            ax2.axvline(10830.1, color='green')
            ax1.set_ylabel('Stellar Spectrum')
            ax2.set_ylabel('Normalized Stellar Spectrum')
            ax2.set_xlabel('Wavelength ($\AA$)')
            ax1.set_xlim(wav_ranges[0][0]-5, wav_ranges[-1][1]+5)
            ax1.legend()
            fig.subplots_adjust(hspace=0)
            plt.show()

    #RASSINE
    ##Making the normalized spectra and error bars.
    ##Making the master normalized spectrum.
    print('RASSINE')
    if Rassine:
        if np.sum(mode=='A') != len(mode) and np.sum(mode=='E')!= len(mode):
            #HA
            for name in os.listdir(rassine_directory+'-HA/MASTER/'):
                if name[:7] == 'RASSINE':
                    obj = pd.read_pickle(rassine_directory+'-HA/MASTER/'+name)
                    norm_master_spctr_HA = obj['flux']/obj['matching_diff']['continuum_linear']
                    master_wav_HA = obj['wave']

            #Extract the Rassine normalized spectra.
            rassine_total_norm_spctr_HA = np.zeros((int(len(os.listdir(rassine_directory+'-HA/STACKED'))/2), len(master_wav_HA)))
            rassine_total_norm_err_HA = np.zeros((int(len(os.listdir(rassine_directory+'-HA/STACKED'))/2), len(master_wav_HA)))
            rassine_total_lamda_HA = np.zeros((int(len(os.listdir(rassine_directory+'-HA/STACKED'))/2), len(master_wav_HA)))
            for l in range(int(len(os.listdir(rassine_directory+'-HA/STACKED'))/2)):
                obj = pd.read_pickle(rassine_directory+'-HA/STACKED/RASSINE_prepared_rassine_'+str(l)+'.p')
                rassine_total_lamda_HA[l] = obj['wave']
                rassine_total_norm_spctr_HA[l] = obj['flux']/obj['matching_diff']['continuum_linear']
                interpol_err = interp1d(total_lamda[l], total_norm_err[l]**2, kind='cubic', fill_value='extrapolate')
                rassine_total_norm_err_HA[l] = np.sqrt(interpol_err(rassine_total_lamda_HA[l]))
            
            #HE
            for name in os.listdir(rassine_directory+'-HE/MASTER/'):
                if name[:7] == 'RASSINE':
                    obj = pd.read_pickle(rassine_directory+'-HE/MASTER/'+name)
                    norm_master_spctr_HE = obj['flux']/obj['matching_diff']['continuum_linear']
                    master_wav_HE = obj['wave']

            #Extract the Rassine normalized spectra.
            rassine_total_norm_spctr_HE = np.zeros((int(len(os.listdir(rassine_directory+'-HE/STACKED'))/2), len(master_wav_HE)))
            rassine_total_norm_err_HE = np.zeros((int(len(os.listdir(rassine_directory+'-HE/STACKED'))/2), len(master_wav_HE)))
            rassine_total_lamda_HE = np.zeros((int(len(os.listdir(rassine_directory+'-HE/STACKED'))/2), len(master_wav_HE)))
            for l in range(int(len(os.listdir(rassine_directory+'-HE/STACKED'))/2)):
                obj = pd.read_pickle(rassine_directory+'-HE/STACKED/RASSINE_prepared_rassine_'+str(l)+'.p')
                rassine_total_lamda_HE[l] = obj['wave']
                rassine_total_norm_spctr_HE[l] = obj['flux']/obj['matching_diff']['continuum_linear']
                interpol_err = interp1d(total_lamda[l], total_norm_err[l]**2, kind='cubic', fill_value='extrapolate')
                rassine_total_norm_err_HE[l] = np.sqrt(interpol_err(rassine_total_lamda_HE[l]))
            
            return total_lamda, rassine_total_lamda_HA, rassine_total_lamda_HE, master_wav_HA, master_wav_HE, total_spctr, total_norm_spctr, rassine_total_norm_spctr_HA, rassine_total_norm_spctr_HE, norm_master_spctr_HA, norm_master_spctr_HE, total_err, total_norm_err, rassine_total_norm_err_HA, rassine_total_norm_err_HE, total_SNR, mode, date, total_RV, total_RV_err, total_FWHM, total_FWHM_err, total_BIS_SPAN, total_BIS_SPAN_err, total_CONTRAST, total_CONTRAST_err, total_H2O, total_H2O_err, total_O2, total_O2_err, total_CO2, total_CO2_err, total_AIRM, total_telluric

        else:
            for name in os.listdir(rassine_directory+'/MASTER/'):
                if name[:7] == 'RASSINE':
                    obj = pd.read_pickle(rassine_directory+'/MASTER/'+name)
                    norm_master_spctr = obj['flux']/obj['matching_diff']['continuum_linear']
                    master_wav = obj['wave']

            #Extract the Rassine normalized spectra.
            rassine_total_norm_spctr = np.zeros((int(len(os.listdir(rassine_directory+'/STACKED'))/2), len(master_wav)))
            rassine_total_norm_err = np.zeros((int(len(os.listdir(rassine_directory+'/STACKED'))/2), len(master_wav)))
            rassine_total_lamda = np.zeros((int(len(os.listdir(rassine_directory+'/STACKED'))/2), len(master_wav)))
            for l in range(int(len(os.listdir(rassine_directory+'/STACKED'))/2)):
                obj = pd.read_pickle(rassine_directory+'/STACKED/RASSINE_prepared_rassine_'+str(l)+'.p')
                rassine_total_lamda[l] = obj['wave']
                rassine_total_norm_spctr[l] = obj['flux']/obj['matching_diff']['continuum_linear']
                interpol_err = interp1d(total_lamda[i], total_norm_err[i]**2, kind='cubic', fill_value='extrapolate')
                rassine_total_norm_err[l] = np.sqrt(interpol_err(master_wav))

            return total_lamda, rassine_total_lamda, master_wav, total_spctr, total_norm_spctr, rassine_total_norm_spctr, norm_master_spctr, total_err, total_norm_err, rassine_total_norm_err, total_SNR, mode, date, total_RV, total_RV_err, total_FWHM, total_FWHM_err, total_BIS_SPAN, total_BIS_SPAN_err, total_CONTRAST, total_CONTRAST_err, total_H2O, total_H2O_err, total_O2, total_O2_err, total_CO2, total_CO2_err, total_AIRM, total_telluric
  
    else:
        return total_lamda, total_spctr, total_norm_spctr, total_err, total_norm_err, total_SNR, mode, date, total_RV, total_RV_err, total_FWHM, total_FWHM_err, total_BIS_SPAN, total_BIS_SPAN_err, total_CONTRAST, total_CONTRAST_err, total_H2O, total_H2O_err, total_O2, total_O2_err, total_CO2, total_CO2_err, total_AIRM, total_telluric
           

def phasefold(t, T, nu):
    '''
    Phase folding function.
    Parameters
    ----------
    :param t: array, time values we want to phase fold.
    :param T: float, reference time stamp used to phase fold.
    :param nu: float, frequency of the phase folding.
    Returns
    :param ph_t: array, phase-folded time array. 
    ----------
    '''
    A = (t-T)*nu
    ph_t = A%1
    return ph_t


def eval_stat(best_fit_theta, x, y, fit_function, param_name):
    '''
    Function to evaluate the statistical significance of the results of a fit.
    Parameters
    ----------
    :param best_fit_theta: array, best-fit parameters from the fit.
    :param x: array, x values of data that are fit.
    :param y: array, y values of data that are fit.
    :param fit_function: function, used to fit the (x, y) data.
    :param param_name: list, containing strings of the name of all the parameters in fit_function. 
    Returns
    ----------
    '''
    #Defining function to evaluate the best-fit
    def f_wrapper_for_odr(beta, t):
        return fit_function(t, *beta)

    Model = so.odrpack.Model(f_wrapper_for_odr)
    data = so.odrpack.Data(x, y)
    myodr = so.odrpack.ODR(data, Model, beta0=best_fit_theta,  maxit=0)
    myodr.set_job(fit_type=2)
    parameterStatistics = myodr.run()
    df_e = len(x) - len(best_fit_theta) # degrees of freedom
    cov_beta = parameterStatistics.cov_beta # parameter covariance matrix from ODR
    sd_beta = parameterStatistics.sd_beta * parameterStatistics.sd_beta
    t_df = ss.t.ppf(0.95, df_e)
    ci = []
    for h in range(len(best_fit_theta)):
        ci.append([best_fit_theta[h] - t_df * parameterStatistics.sd_beta[h], best_fit_theta[h] + t_df * parameterStatistics.sd_beta[h]])

    tstat_beta = best_fit_theta / parameterStatistics.sd_beta # coeff t-statistics
    pstat_beta = (1.0 - ss.t.cdf(np.abs(tstat_beta), df_e)) * 2.0    # coef. p-values

    for j in range(len(best_fit_theta)):
        print('parameter:', param_name[j]+':', best_fit_theta[j])
        print('   95% conf interval:', ci[j][0], ci[j][1])
        print('   tstat:', tstat_beta[j])
        print('   pstat:', pstat_beta[j])
