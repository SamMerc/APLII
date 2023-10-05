import numpy as np
from scipy.special import voigt_profile, wofz
import matplotlib.pyplot as plt


elec_ch= 1.6021766208e-19   # A . s or C
c_light_m=299792458.    # m / s
m_e = 9.10938356e-31    # kg
Eps_0 = 8.854187817620e-12  # A^2 s^4 kg−1 m−3
JOULE_TO_CM1=5.040963080525957e+22

factor = elec_ch**2/(4*Eps_0*m_e* c_light_m)    # m**2 / s
#factor = np.pi * elec_ch**2  /(m_e * c) # C**2 . s / (kg . m)

amu=1.660531e-27    # kg
m_he = 4.002602 * amu
m_si = 28.0855 * amu

k_boltz=1.3806488e-23   # m2 kg s-2 K-1
h_planck=6.62606957e-34  # kg m2 s-1

Aki_He = 1.0216e+07
Aki_Si = 1.97e+07 # s-1

f_osc_si = 3.47e-1
f_oscHe = {'He1' : 5.9902e-02,'He2': 1.7974e-01,'He3': 2.9958e-01}
	

ref_wav_He_vacuum={'He1':10832.057472, 'He2':10833.216751, 'He3':10833.306444} # angstrom

ref_wav_He_air={'He1':10829.09114, 'He2':10830.25010, 'He3':10830.33977}

ref_wav_Si_vacuum = 10830.0549 # angstrom

ref_wav_Si_air = 10827.089

x0_he={ 'He1':-32.08124686382154, 'He2':0, 'He3': 2.4821145512899165} # km/s

def He_triplet_line_wav(x, T,n_col, v_sini,delta_d, choice, medium, R_pow, micro_t, RV_offset):

    '''
    
    - medium is either air ou vacuum. This depend on the data you are working with.

    - choice is to choose which He line to include.
        By default the three lines are used.
        
    '''

    return_value = 1.

    if medium == "AIR":
        if choice[0]:

            return_value *= abs_line_wav(x,f_oscHe['He1'],T, n_col, v_sini, m_he, ref_wav_He_air['He1'],delta_d, Aki_He, R_pow, micro_t, RV_offset)

        if choice[1] :

            return_value *= abs_line_wav(x,f_oscHe['He2'],T, n_col, v_sini, m_he, ref_wav_He_air['He2'],delta_d, Aki_He, R_pow, micro_t, RV_offset)

        if choice[2] :

            return_value *= abs_line_wav(x,f_oscHe['He3'],T, n_col, v_sini, m_he, ref_wav_He_air['He3'],delta_d, Aki_He, R_pow, micro_t, RV_offset)

    elif medium =="VACUUM":
        if choice[0]:

            return_value *= abs_line_wav(x,f_oscHe['He1'],T, n_col, v_sini, m_he, ref_wav_He_vacuum['He1'],delta_d, Aki_He, R_pow, micro_t, RV_offset)

        if choice[1] :

            return_value *= abs_line_wav(x,f_oscHe['He2'],T, n_col, v_sini, m_he, ref_wav_He_vacuum['He2'],delta_d, Aki_He, R_pow, micro_t, RV_offset)

        if choice[2] :

            return_value *= abs_line_wav(x,f_oscHe['He3'],T, n_col, v_sini, m_he, ref_wav_He_vacuum['He3'],delta_d, Aki_He, R_pow, micro_t, RV_offset)

    return return_value

def Si_line_wav(x, T,n_col, v_sini,delta_d, medium, R_pow, micro_t, RV_offset):

    '''
    
    - medium is either air ou vacuum. This depend on the data you are working with.

    - choice is to choose which He line to include.
        By default the three lines are used.
        
    '''

    return_value = 1.

    if medium == "AIR":
        return_value *= abs_line_wav(x,f_osc_si,T, n_col, v_sini, m_si, ref_wav_Si_air,delta_d, Aki_Si, R_pow, micro_t, RV_offset)

    elif medium =="VACUUM":
        return_value *= abs_line_wav(x,f_osc_si,T, n_col, v_sini, m_si, ref_wav_Si_vacuum,delta_d, Aki_Si, R_pow, micro_t, RV_offset)

    return return_value

def He_triplet_and_Si_line_wav(x, T_He, T_Si, n_col_He, n_col_Si, v_sini, delta_d_He, delta_d_Si, choice, medium, R_pow, micro_t, RV_offset_He, RV_offset_Si):
    '''
    
    - medium is either air ou vacuum. This depend on the data you are working with.

    - choice is to choose which He line to include.
        By default the three lines are used.
        
    '''

    return_value = 1.

    if medium == "AIR":
        if choice[0]:

            return_value *= abs_line_wav(x,f_oscHe['He1'],T_He, n_col_He, v_sini, m_he, ref_wav_He_air['He1'],delta_d_He, Aki_He, R_pow, micro_t, RV_offset_He)

        if choice[1] :

            return_value *= abs_line_wav(x,f_oscHe['He2'],T_He, n_col_He, v_sini, m_he, ref_wav_He_air['He2'],delta_d_He, Aki_He, R_pow, micro_t, RV_offset_He)

        if choice[2] :

            return_value *= abs_line_wav(x,f_oscHe['He3'],T_He, n_col_He, v_sini, m_he, ref_wav_He_air['He3'],delta_d_He, Aki_He, R_pow, micro_t, RV_offset_He)
            
        if choice[3] :

            return_value *= abs_line_wav(x, f_osc_si, T_Si, n_col_Si, v_sini, m_si, ref_wav_Si_air, delta_d_Si, Aki_Si, R_pow, micro_t, RV_offset_Si)
        

    elif medium =="VACUUM":
        if choice[0]:

            return_value *= abs_line_wav(x,f_oscHe['He1'],T_He, n_col_He, v_sini, m_he, ref_wav_He_vacuum['He1'],delta_d_He, Aki_He, R_pow, micro_t, RV_offset_He)

        if choice[1] :

            return_value *= abs_line_wav(x,f_oscHe['He2'],T_He, n_col_He, v_sini, m_he, ref_wav_He_vacuum['He2'],delta_d_He, Aki_He, R_pow, micro_t, RV_offset_He)

        if choice[2] :

            return_value *= abs_line_wav(x,f_oscHe['He3'],T_He, n_col_He, v_sini, m_he, ref_wav_He_vacuum['He3'],delta_d_He, Aki_He, R_pow, micro_t, RV_offset_He)

        if choice[3] :

            return_value *= abs_line_wav(x, f_osc_si, T_Si, n_col_Si, v_sini, m_si, ref_wav_Si_vacuum, delta_d_Si, Aki_Si, R_pow, micro_t, RV_offset_Si)
            
    return return_value
def He_triplet_and_Si_line_wav_Complex(x, T_He, T_Si, n_col_He, n_col_Si, v_sini, delta_d_He, delta_d_Si, choice, medium, R_pow, micro_t, RV_offset_He, RV_offset_Si):
    '''
    
    - medium is either air ou vacuum. This depend on the data you are working with.

    - choice is to choose which He line to include.
        By default the three lines are used.
        
    '''

    return_value = 1.

    if medium == "AIR":
        if choice[0]:

            return_value *= abs_line_wav(x,f_oscHe['He1'],T_He, n_col_He, v_sini, m_he, ref_wav_He_air['He1'],delta_d_He, Aki_He, R_pow, micro_t, RV_offset_He)

        if choice[1] :

            return_value *= abs_line_wav(x,f_oscHe['He2'],T_He, n_col_He, v_sini, m_he, ref_wav_He_air['He2'],delta_d_He, Aki_He, R_pow, micro_t, RV_offset_He)

        if choice[2] :

            return_value *= abs_line_wav(x,f_oscHe['He3'],T_He, n_col_He, v_sini, m_he, ref_wav_He_air['He3'],delta_d_He, Aki_He, R_pow, micro_t, RV_offset_He)
            
        if choice[3] :

            return_value *= abs_line_wav(x, f_osc_si, T_Si, n_col_Si, v_sini, m_si, ref_wav_Si_air, delta_d_Si, Aki_Si, R_pow, micro_t, RV_offset_Si)
        

    elif medium =="VACUUM":
        if choice[0]:

            return_value *= abs_line_wav(x,f_oscHe['He1'],T_He, n_col_He, v_sini, m_he, ref_wav_He_vacuum['He1'],delta_d_He, Aki_He, R_pow, micro_t, RV_offset_He)

        if choice[1] :

            return_value *= abs_line_wav(x,f_oscHe['He2'],T_He, n_col_He, v_sini, m_he, ref_wav_He_vacuum['He2'],delta_d_He, Aki_He, R_pow, micro_t, RV_offset_He)

        if choice[2] :

            return_value *= abs_line_wav(x,f_oscHe['He3'],T_He, n_col_He, v_sini, m_he, ref_wav_He_vacuum['He3'],delta_d_He, Aki_He, R_pow, micro_t, RV_offset_He)

        if choice[3] :

            return_value *= abs_line_wav(x, f_osc_si, T_Si, n_col_Si, v_sini, m_si, ref_wav_Si_vacuum, delta_d_Si, Aki_Si, R_pow, micro_t, RV_offset_Si)
            
    return return_value

def abs_line_wav(x,f_osc, T, n_col, v_sini, m, lambda0,delta_d, Aki, R_pow, micro_t, RV_offset):

    '''
    
    Function to build an absorption line.
    Either use Gaussian profile or Voigt profile.
    Decided with the value of d, d = 0 for a Gaussian.
    d != 0 for a Voigt.
    
    Arguments:
    - x = wavelength vector [e-10 m]

    - f_osc is the oscillator stength

    - T for temperature [K]

    - n_col is column density

    - vsini of star in [km s^-1]

    - m mass of the element

    - lambda0 is the transition wavelength [e-10 m]

    - d is the damping factor = delta_d * lambda0 * 1e-10 * Aki / (4.0 * np.pi * width).
            Leave to 0 to have Gaussian profiles. Set to a value greater than 0 to have Voigt profiles
    
    - Aki in s-1 Einstein coefficient
    
    - Instrumental resolving power
    
    - microturbulence in [km s^-1]

    '''

    #########################
    # Broadening  non thermal
    #########################
    broad_NT = 0.
    
    # Velocity broadening
    if v_sini is not None:
        broad_NT += (v_sini*1e3)**2
    
    # Instrumental broadening
    if R_pow is not None:
        broad_NT += (c_light_m/R_pow)**2
    
    # micro turbulence
    if micro_t is not None:
        broad_NT += (micro_t*1e3)**2
    
    ####################
    # Thermal broadening
    vt = np.sqrt(2.*k_boltz * T / m)

    
    width = np.sqrt(vt**2 + broad_NT)
    
    d = delta_d * lambda0 * 1e-10 * Aki/(4*np.pi * width)
    
    if d !=0:
    # Voigt

        # With d  = damping / width, avec damping dans l'unité de width
        # we specify what fraction of width the dmaping is with d
        
        z = ((c_light_m/ lambda0 * (x - lambda0)) + RV_offset) / (width) + 1j*d
        
        line_profile = wofz(z).real

    else :
    # Gaussian
    
        line_profile = np.exp( - ( (c_light_m/lambda0 * (x - lambda0))+RV_offset )**2 / (width**2))
    
    return np.exp(- factor * f_osc * lambda0 * 1e-10 / (np.sqrt(np.pi) * width) * n_col * line_profile)


