
import numpy as np
from line_model import *
from functions import *
from copy import deepcopy


def spectrum_he(wavelength, I,  He_temperature,He_density, vsini, medium, choice = [True,True,True], He_delta_damping = 0, R_pow = None, micro_t = None) :
    
    # Your spectrum
    Intensity = deepcopy(I)


    # Only absorption, no emission
    etau = He_triplet_line_wav(wavelength, He_temperature, He_density, vsini, He_delta_damping, choice, medium, R_pow, micro_t)
    snu = 0.0

        
        
    return Intensity * etau + snu * (1.0 - etau)

