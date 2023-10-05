
import numpy as np
import line_model as lm
from importlib import reload
from functions import *
from copy import deepcopy
reload(lm)

def spectrum_he(wavelength, I,  He_temperature,He_density, vsini, medium, RV_offset, choice = [True,True,True], He_delta_damping = 0, R_pow = None, micro_t = None) :
    
    # Your spectrum
    Intensity = deepcopy(I)


    # Only absorption, no emission
    etau = lm.He_triplet_line_wav(wavelength, He_temperature, He_density, vsini, He_delta_damping, choice, medium, R_pow, micro_t, RV_offset)
    snu = 0.0

        
        
    return Intensity * etau + snu * (1.0 - etau)

def spectrum_Si(wavelength, I,  Si_temperature,Si_density, vsini, medium, RV_offset_Si, Si_delta_damping = 0, R_pow = None, micro_t = None) :
    
    # Your spectrum
    Intensity = deepcopy(I)


    # Only absorption, no emission
    etau = lm.Si_line_wav(wavelength, Si_temperature, Si_density, vsini, Si_delta_damping, medium, R_pow, micro_t, RV_offset_Si)
    snu = 0.0

        
        
    return Intensity * etau + snu * (1.0 - etau)

def spectrum_he_and_Si(wavelength, I,  He_temperature, Si_temperature, He_density, Si_density, vsini, medium, Si_delta_damping, RV_offset_He, RV_offset_Si, He_delta_damping = 0, R_pow = None, choice = [True,True,True,True], micro_t = None) :
    
    # Your spectrum
    Intensity = deepcopy(I)


    # Only absorption, no emission
    etau = lm.He_triplet_and_Si_line_wav(wavelength, He_temperature, Si_temperature, He_density, Si_density, vsini, He_delta_damping, Si_delta_damping, choice, medium, R_pow, micro_t, RV_offset_He, RV_offset_Si)
    snu = 0.0

        
        
    return Intensity * etau + snu * (1.0 - etau)

def spectrum_he_and_Si_Gauss(wavelength, I,  He_temperature, Si_temperature, He_density, Si_density, vsini, medium, RV_offset_He, RV_offset_Si, He_delta_damping = 0, Si_delta_damping=0, R_pow = None, choice = [True,True,True,True], micro_t = None) :
    
    # Your spectrum
    Intensity = deepcopy(I)


    # Only absorption, no emission
    etau = lm.He_triplet_and_Si_line_wav(wavelength, He_temperature, Si_temperature, He_density, Si_density, vsini, He_delta_damping, Si_delta_damping, choice, medium, R_pow, micro_t, RV_offset_He, RV_offset_Si)
    snu = 0.0

        
        
    return Intensity * etau + snu * (1.0 - etau)


def spectrum_he_and_Si_Complex(wavelength, I,  He_temperature, Si_temperature, He_density, Si_density, vsini, medium, RV_offset_He, RV_offset_Si, He_delta_damping = 0, Si_delta_damping=0, R_pow = None, choice = [True,True,True,True], micro_t = None) :
    
    # Your spectrum
    Intensity = deepcopy(I)


    # Only absorption, no emission
    etau = lm.He_triplet_and_Si_line_wav(wavelength, He_temperature, Si_temperature, He_density, Si_density, vsini, He_delta_damping, Si_delta_damping, choice, medium, R_pow, micro_t, RV_offset_He, RV_offset_Si)
    snu = 0.0

        
        
    return Intensity * etau + snu * (1.0 - etau)

