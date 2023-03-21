import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pf
import os
from astropy.modeling import models
from astropy import units as u
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle
from scipy.special import voigt_profile


def extraction(file_directory, blaze_directory, CCF_directory, order):
    
    #Initialize the arrays
    #Contains the wavelength for the order of interest for the spectra
    total_lamda = np.zeros((len(os.listdir(file_directory)), 4084))
    
    #Contains the flux and normalized values of the spectra of the order of interest
    total_spctr = np.zeros((len(os.listdir(file_directory)), 4084))
    total_norm_spctr = np.zeros((len(os.listdir(file_directory)), 4084))
    
    #Contains the error on the flux and normalized flux values of the spectra
    total_err = np.zeros((len(os.listdir(file_directory)), 4084))
    total_norm_err = np.zeros((len(os.listdir(file_directory)), 4084))
    
    #Contains the SNR value of the spectra
    total_SNR = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the mode of each spectrum
    mode = np.zeros((len(os.listdir(file_directory))), dtype=str)
    
    #Contains the timestamp at which each spectrum was collected (?)
    date = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the RV values of the spectra
    #The RV values are obtained with a CCF routine, that has been TC
    total_RV = np.zeros((len(os.listdir(file_directory))))
    
    #Error on the RV values of the spectra
    total_RV_err = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the FWHM of the CCF of the spectra
    total_FWHM = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the Bisector span of the CCF of the spectra
    total_BIS_SPAN = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the integrated column density of H2O, CO2 and O2 at the time of acquisition of the spectra
    total_H2O = np.zeros((len(os.listdir(file_directory))))
    total_O2 = np.zeros((len(os.listdir(file_directory))))
    total_CO2 = np.zeros((len(os.listdir(file_directory))))
    
    #Contains the airmass at the time of observation of each spectrum
    total_AIRM = np.zeros((len(os.listdir(file_directory))))
    
    #Defining the Sun's BB for later
    Sun_BB = models.BlackBody(temperature = 5778*u.K)

    #Initial mode analysis to see the modes used in the data
    for i in range(len(os.listdir(file_directory))):
        file = pf.open(file_directory+'/'+os.listdir(file_directory)[i])
        mode[i] = file[0].header['HIERARCH ESO INS MODE'][1]
    
    for i in range(len(os.listdir(file_directory))):
        #Opening the files in the directories
        file = pf.open(file_directory+'/'+os.listdir(file_directory)[i])
        file_CCF = pf.open(CCF_directory +'/'+os.listdir(CCF_directory)[i])
        
        #Extracting the wavelength
        total_lamda[i] = file[4].data[order]

        #Extracting the DLL for spectrum correction - only used later
        file_DLL = file[6].data[order]
        
        #Extracting the timestamp of each spectra
        date[i] = file[0].header['MJD-OBS']

        #Getting the RV and error on the RV from the CCF files
        total_RV[i] = file_CCF[0].header['HIERARCH ESO QC CCF RV']
        total_RV_err[i] = file_CCF[0].header['HIERARCH ESO QC CCF RV ERROR']
        
        #Getting the FWHM
        total_FWHM[i] = file[0].header['HIERARCH ESO QC CCF FWHM']
    
        #Getting the Bis Span
        total_BIS_SPAN[i] = file[0].header['HIERARCH ESO QC CCF BIS SPAN']
    
        #Getting the airmass 
        total_AIRM[i] = (file[0].header['HIERARCH ESO TEL AIRM START'] + file[0].header['HIERARCH ESO TEL AIRM END'])/2
    
        #Getting the integrated column density for H2O, O2, CO2
        total_H2O[i] = file[0].header['HIERARCH ESO QC TELL H2O IWV']
        total_O2[i] = file[0].header['HIERARCH ESO QC TELL O2 IWV']
        total_CO2[i] = file[0].header['HIERARCH ESO QC TELL CO2 IWV']
    
        #Retrieving the SNR of each spectra at the order of interest
        total_SNR[i] = file[0].header['HIERARCH ESO QC ORDER'+str(order)+' SNR']
        
        file_spctr = file[8].data[order]
        file_err = file[10].data[order]

        #If there are two modes of observation
        if np.sum(mode=='A') != len(mode):
            #Looking through the Blaze directory to find the Blaze files for each observation mode
            for j in range(len(os.listdir(blaze_directory))):
                file_blaze = pf.open(blaze_directory+'/'+os.listdir(blaze_directory)[j])
                if file_blaze[0].header['HIERARCH ESO INS MODE'][1] == 'A':
                    blaze_HA = file_blaze
                else:
                    blaze_HE = file_blaze
                         
            blaze_HA_spctr = blaze_HA[1].data[order]
            blaze_HE_spctr = blaze_HE[1].data[order]
            
            #Performing the Blaze correction -- the Blaze file used depends on the observation mode used

            if mode[i] == 'A':
                BC_spctr = file_spctr/blaze_HA_spctr
                BC_err = file_err/blaze_HA_spctr
            else:
                BC_spctr = file_spctr/blaze_HE_spctr
                BC_err = file_err/blaze_HE_spctr

            #Removing the trend of the spectrum's continuum using DLL
            BC_DLL_spctr = BC_spctr/file_DLL
            BC_DLL_err = BC_err/file_DLL
            
            total_spctr[i] = BC_DLL_spctr/Sun_BB(total_lamda[i]*u.AA).value
            total_err[i] = BC_DLL_err/Sun_BB(total_lamda[i]*u.AA).value
            
            total_norm_spctr[i] = total_spctr[i]/np.mean(total_spctr[i])
            total_norm_err[i] = total_err[i]/np.mean(total_spctr[i])
                    
        #If there is only one mode of observation
        else:
            blaze = pf.open(blaze_directory+'/'+os.listdir(blaze_directory)[0])
            
            blaze_spctr = blaze[1].data[order]
            
            #Performing the Blaze correction
            BC_spctr = file_spctr/blaze_HE_spctr
            BC_err = file_err/blaze_HE_spctr
            
            #Removing the trend of the spectrum's continuum using DLL and the Sun's Black Body spectrum
            BC_DLL_spctr = BC_spctr/file_DLL
            BC_DLL_err = BC_err/file_DLL
            
            total_spctr[i] = BC_DLL_spctr/Sun_BB(file[4].data[order]*u.AA).value
            total_err[i] = BC_DLL_err/Sun_BB(file[4].data[order]*u.AA).value
            
            total_norm_spctr[i] = total_spctr[i]/np.mean(total_spctr[i])
            total_norm_err[i] = total_err[i]/np.mean(total_spctr[i])

    return total_lamda, total_spctr, total_norm_spctr, total_err, total_norm_err, total_SNR, mode, date, total_RV, total_RV_err, total_FWHM, total_BIS_SPAN, total_H2O, total_O2, total_CO2, total_AIRM




def segment_and_reduce(modes, SNR, L):
    #We separate the spectra based on their observing mode
    L_HA = L[modes=='A']
    L_HE = L[modes=='E']
    
        #We cutoff spectra with SNR lower than 300. 
        #We notice that this is only the case for spectra obtained
        #with the HA observing method.
        
    SNR_HA = SNR[modes=='A']
    SNR_HE = SNR[modes=='E']
    
    L_HA = L_HA[SNR_HA>300]
        
    return L_HA, L_HE



def vac2air(wavelength):
# Transform wavelength in vacuum to air

# See VALD website here

# http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

# The formula comes from Birch and Downs (1994, Metrologia, 31, 315)

    s = 1e4 / wavelength # s = 10^4 / lamda_vac, lamda_vac is in Angstrom

    n_air = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)

    return wavelength / n_air



def air2vac(wavelength):
# Transform wavelength in vacuum to air

# See VALD website here

# http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

# The formula comes from Birch and Downs (1994, Metrologia, 31, 315)

    s = 1e4 / wavelength # s = 10^4 / lamda_air, lamda_air is in Angstrom

    n_vac = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)

    return wavelength * n_vac

