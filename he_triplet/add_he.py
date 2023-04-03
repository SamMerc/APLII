#!/usr/bin/env python3

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from spectrum_model import *
from functions import *

###################################################
###                                             ###
###               Reading files                 ###
###                                             ###
###################################################

###################################################
###                                             ###
### Observed spectrum                           ###
###                                             ###
###################################################

#########
# Example
# Should come from a file that contains your observed data
# wavelength is your wavelength vector in A
wavelength_obs = np.arange(10820,10840,0.01)
# flux is your flux vector
flux_obs = np.ones(np.shape(wavelength_obs))

# You can check your observed spectrum
plt.plot(wavelength_obs,flux_obs)
plt.show()



###################################################
###                                             ###
###           end of file reading               ###
###                                             ###
###################################################


# Helium density
density = 1.15e15

# Helium temperature
temperature = 10000

# Stellar rotation velocity
vsini = 2

# Medium of observations
#medium="AIR"
medium= "VACUUM"

# Instrumental resolving power e.g. R_pow = 70000
# For non thermal broadening
# see spectrum_he() in spectrum_model.py to know where to write the argument
#R_pow# = 7000

# Microturbulence
# for non thermal broadening
# micro_t = 1

# You can change the damping to have Voigt profiles, by default you have Gaussian profiles
# Set to value greater than 0
# d_damping = 0


# see spectrum_he() in spectrum_model.py to know where to write the above argument when call the function


#######################################################
#######################################################
#######################################################

flux_obs_he=spectrum_he(wavelength_obs, flux_obs,  temperature,density, vsini, medium)

#######################################################
#######################################################
#######################################################



fig,ax = plt.subplots(figsize=(10,7))

custom_axis(plt,ax=ax,dir_x='out',dir_y='out',xmajor_length=6,ymajor_length=6,
            xmajor_int=5,xminor_int=2.5,ymajor_int=0.1,yminor_int=0.1,
            xmajor_form='%i',ymajor_form='%1.2f',
            #x_title=r'Wavelength $\AA$',y_title='Normalised flux',
            font_size=12,xfont_size=12,yfont_size=12)



ax.plot(wavelength_obs, flux_obs)
ax.plot(wavelength_obs, flux_obs_he)

plt.xlabel(r'Wavelength ($\AA$)')
#plt.ylabel('Normalised flux')

#ax.set_ylim(0.38,1.12)
#plt.legend()
plt.show()
    





