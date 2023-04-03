from __future__ import print_function, division
import numpy as np
from PyAstronomy import pyasl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.colors as colors






# Used to convert from vacuum/air wavelength to air/vacuum wavelength
def convert_wavelength(wvl, medium ):
#https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/pyasl_wvlconv.html
        if medium == "vac":
            # Convert wavelength in air to wavelength in vacuum in angstrom.
            # By default, the conversion specified by Ciddor 1996 are used.
            wvlVac = pyasl.airtovac2(wvl)
            print("Wavelength in vacuum: ", wvlVac)
            return wvlVac

        if medium =="air":
            # Convert wavelength from vacuum to air
            wvlAir = pyasl.vactoair2(wvl)
            print("Wavelength in air: ", wvlAir)
            return wvlAir


    
def np_where1D(cond):
    return np.where(cond)[0]

# To convert wavelength to km s^-1
# l is the wavelength vector angstrom
# r is the reference wavelength of the line to be at 0 km/s
def lambdatokms(l,r):
    c_light_m=299792458.
    return c_light_m*(l/r-1.)/1000.
    
# To convert km/s to wavelength
# v is the velocity vector
# ref_l is the reference wavelength in angstrom of the line at 0 km/s
def kmstolambda(v,ref_l):
    c_light_m=299792458.
    return ((v*1000./c_light_m+1.)*ref_l)*1e10
'''
Routine to define axis properties
'''
def custom_axis(plt,ax=None,x_range=None,y_range=None,z_range=None,position=None,colback=None,
            x_mode=None,y_mode=None,z_mode=None,
            x_title=None,y_title=None,z_title=None,x_title_dist=None,y_title_dist=None,z_title_dist=None,
            no_xticks=False,no_yticks=False,no_zticks=True,
            top_xticks=None,right_yticks=None,
            xmajor_int=None,xminor_int=None,ymajor_int=None,yminor_int=None,zmajor_int=None,zminor_int=None,
            xmajor_form=None,ymajor_form=None,zmajor_form=None,
            xmajor_length=None,ymajor_length=None,zmajor_length=None,xminor_length=None,yminor_length=None,zminor_length=None,
            font_size=None,font_thick=None,xfont_size=None,yfont_size=None,zfont_size=None,
            axis_thick=None,xmajor_thick=None,xminor_thick=None,ymajor_thick=None,yminor_thick=None,zmajor_thick=None,zminor_thick=None,
            dir_x=None,dir_y=None,dir_z=None,
                xtick_pad=None,ytick_pad=None,ztick_pad=None,
                xlab_col=None,ylab_col=None,zlab_col=None,
                hide_axis=None,
                right_axis=False,secy_title=None,secy_range=None,secy_title_dist=None,no_secyticks=None,secymajor_int=None,dir_secy=None,secyfont_size=None,
                secymajor_length=None,secymajor_thick=None,secyminor_length=None,secyminor_thick=None,secymajor_form=None,secyminor_int=None,secylab_col=None):
                            
#TODO: en fait on peut ajouter dans une figure des sousplots ou on veut avec
#ax=fig.add_axes([left,right,width,height])    en fraction de 0,1
                            
                    
      #Font
      #    - thick: 'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
    font_size_loc=font_size if font_size is not None else 10.
    ## William: defined in plot_simu.py line 57
    font_thick_loc=font_thick if font_thick is not None else 'normal'
    ## William: unlocated
    plt.rc('font', size=font_size_loc,weight=font_thick_loc,**{'family':'sans-serif','sans-serif':['Helvetica']})
#    plt.rc('text', usetex=True)

      #Axis
    if ax==None:ax=plt.gca()

    #Axis frame position
    #    - corresponds to the corners of the image
    if position is not None:
        plt.subplots_adjust(left=position[0],bottom=position[1],right=position[2],top=position[3])

      #Set axis to log mode if required
    if x_mode=='log':ax.set_xscale('log')
    if y_mode=='log':ax.set_yscale('log')

      #Axis ranges
    if x_range is not None:ax.set_xlim([x_range[0], x_range[1]])
    if y_range is not None:ax.set_ylim([y_range[0], y_range[1]])
    if z_range is not None:ax.set_zlim([z_range[0], z_range[1]])

      #Axis titles
    xfont_size_loc=xfont_size if xfont_size is not None else 10.
    yfont_size_loc=yfont_size if yfont_size is not None else 10.
    zfont_size_loc=zfont_size if zfont_size is not None else 10.
    if x_title is not None:
            ax.set_xlabel(x_title,fontsize=xfont_size_loc,weight=font_thick_loc)
    if y_title is not None:
            if (right_yticks=='on'):  #set title to right axis
                ax.set_ylabel(y_title,fontsize=yfont_size_loc,rotation=270,labelpad=22,weight=font_thick_loc)
            else:
                ax.set_ylabel(y_title,fontsize=yfont_size_loc,weight=font_thick_loc)
    if z_title is not None:
            ax.set_zlabel(z_title,fontsize=zfont_size_loc,weight=font_thick_loc)
                
      #Axis title distance
    if x_title_dist is not None:ax.xaxis.labelpad = x_title_dist
    if y_title_dist is not None:ax.yaxis.labelpad = y_title_dist
    if z_title_dist is not None:ax.zaxis.labelpad = z_title_dist
    
      #Axis background color
    if colback is not None:ax.set_facecolor(colback)

      #Axis thickness
    axis_thick_loc=axis_thick if axis_thick is not None else 1.
    if ('bottom' in list(ax.spines.keys())):ax.spines['bottom'].set_linewidth(axis_thick_loc)
    if ('top' in list(ax.spines.keys())):   ax.spines['top'].set_linewidth(axis_thick_loc)
    if ('left' in list(ax.spines.keys())):  ax.spines['left'].set_linewidth(axis_thick_loc)
    if ('right' in list(ax.spines.keys())): ax.spines['right'].set_linewidth(axis_thick_loc)

      #X ticks and label on top
    if (top_xticks=='on'):
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
      #Y ticks and label on right
    if (right_yticks=='on'):
         ax.yaxis.tick_right()
         ax.yaxis.set_label_position('right')

         
    #------------------------------------------------------------------------
    #X ticks (on by default)
    if (no_xticks==False):
        
          #Interval between major ticks
        if xmajor_int is not None:ax.xaxis.set_major_locator(MultipleLocator(xmajor_int))

        #Direction of ticks
        if dir_x==None:dir_x='in'

          #Major ticks
        xmajor_length_loc=xmajor_length if xmajor_length is not None else 7
        xmajor_thick_loc=xmajor_thick if xmajor_thick is not None else 1.5
        xtick_pad_loc=xtick_pad if xtick_pad is not None else 5
        ax.tick_params('x', length=xmajor_length_loc, which='major',width=xmajor_thick_loc,
                       direction=dir_x, pad=xtick_pad_loc,labelsize=xfont_size_loc,top=True) #labelrotation=30
    
          #Minor ticks
        xminor_length_loc=xminor_length if xminor_length is not None else xmajor_length_loc/2.
        xminor_thick_loc=xminor_thick if xminor_thick is not None else 1.5
        ax.tick_params('x', length=xminor_length_loc, which='minor',width=xminor_thick_loc,
                      direction=dir_x,labelsize=xfont_size_loc,top=True)
    
          #Major ticks label format
        if xmajor_form is not None:ax.xaxis.set_major_formatter(FormatStrFormatter(xmajor_form))
    
          #Interval between minor ticks
        if xminor_int is not None:ax.xaxis.set_minor_locator(MultipleLocator(xminor_int))

          #Ticks labels color
        if xlab_col is not None:[i_col.set_color(xlab_col) for i_col in ax.get_xticklabels()]

    else:
        ax.set_xticks([])
        
    #-----------------
    #Y ticks (on by default)
    if (no_yticks==False):
        
          #Interval between major ticks
        if ymajor_int is not None:ax.yaxis.set_major_locator(MultipleLocator(ymajor_int))

        #Direction of ticks
        if dir_y==None:dir_y='in'

          #Major ticks length
        ymajor_length_loc=ymajor_length if ymajor_length is not None else 7
        ymajor_thick_loc=ymajor_thick if ymajor_thick is not None else 1.5
        ytick_pad_loc=ytick_pad if ytick_pad is not None else 5
        ax.tick_params('y', length=ymajor_length_loc, which='major',width=ymajor_thick_loc,
                      direction=dir_y, pad=ytick_pad_loc,labelsize=yfont_size_loc, right=True)
    
          #Minor ticks length
        yminor_length_loc=yminor_length if yminor_length is not None else ymajor_length_loc/2.
        yminor_thick_loc=yminor_thick if yminor_thick is not None else 1.5
        ax.tick_params('y', length=yminor_length_loc, which='minor',width=yminor_thick_loc,
                      direction=dir_y,labelsize=yfont_size_loc, right=True)
    
          #Major ticks label format
        if ymajor_form is not None:ax.yaxis.set_major_formatter(FormatStrFormatter(ymajor_form))
            
          #Interval between minor ticks
        if yminor_int is not None:ax.yaxis.set_minor_locator(MultipleLocator(yminor_int))

          #Ticks labels color
        if ylab_col is not None:[i_col.set_color(ylab_col) for i_col in ax.get_yticklabels()]

    else:
        ax.set_yticks([])

    #-----------------
    #Z ticks (off by default)
    if (no_zticks==False):
        
          #Interval between major ticks
        if zmajor_int is not None:ax.zaxis.set_major_locator(MultipleLocator(zmajor_int))

        #Direction of ticks
        if dir_z==None:dir_z='in'

          #Major ticks length
        zmajor_length_loc=zmajor_length if zmajor_length is not None else 7
        zmajor_thick_loc=zmajor_thick if zmajor_thick is not None else 1.
        ztick_pad_loc=ztick_pad if ztick_pad is not None else 5
        ax.tick_params('z', length=zmajor_length_loc, which='major',width=zmajor_thick_loc,
                      direction=dir_z, pad=ztick_pad_loc,labelsize=zfont_size_loc)
    
          #Minor ticks length
        zminor_length_loc=zminor_length if zminor_length is not None else zmajor_length_loc/2.
        zminor_thick_loc=zminor_thick if zminor_thick is not None else 1.5
        ax.tick_params('z', length=zminor_length_loc, which='minor',width=zminor_thick_loc,
                      direction=dir_z,labelsize=zfont_size_loc)
    
          #Major ticks label format
        if zmajor_form is not None:
            majorFormatter = FormatStrFormatter(zmajor_form)
            ax.zaxis.set_major_formatter(majorFormatter)
            
          #Interval between minor ticks
        if zminor_int is not None:
            minorLocator   = MultipleLocator(zminor_int)
            ax.zaxis.set_minor_locator(minorLocator)

          #Ticks labels color
        if zlab_col is not None:[i_col.set_color(zlab_col) for i_col in ax.get_zticklabels()]
    #William
        #else:
#ax.set_zticks([])
        
    #------------------------------------------------------------------------
    #Secondary axis (right side)
    if right_axis==True:
         if right_yticks=='on':stop('Nominal axis already set to right axis')
         newaxvert = ax.twinx()
         newaxvert.yaxis.tick_right()
         newaxvert.yaxis.set_label_position('right')

         #Title
         if secy_title is not None:
             yfont_size_loc=secyfont_size if secyfont_size is not None else 10.
             newaxvert.set_ylabel(secy_title,fontsize=yfont_size_loc,rotation=0)
         if secy_title_dist is not None:newaxvert.yaxis.labelpad = secy_title_dist

         #Range
         if secy_range is not None:newaxvert.set_ylim([secy_range[0], secy_range[1]])
    
           #-----------------
         if (no_secyticks==None) or (no_secyticks==''):
             
             #Interval between major ticks
             if secymajor_int is not None:
                newaxvert.yaxis.set_major_locator(MultipleLocator(secymajor_int))
                
             #Direction of ticks
             if dir_secy==None:dir_secy='in'
    
              #Major ticks length
             secymajor_length_loc=secymajor_length if secymajor_length is not None else 10
             secymajor_thick_loc=secymajor_thick if secymajor_thick is not None else 1.5
             newaxvert.tick_params('y', length=secymajor_length_loc, which='major',width=secymajor_thick_loc,
                          direction=dir_secy)
        
              #Minor ticks length
             secyminor_length_loc=secyminor_length if secyminor_length is not None else secymajor_length_loc/2.
             secyminor_thick_loc=secyminor_thick if secyminor_thick is not None else 1.5
             newaxvert.tick_params('y', length=secyminor_length_loc, which='minor',width=secyminor_thick_loc,
                          direction=dir_secy)
        
              #Major ticks label format
             if secymajor_form is not None:newaxvert.yaxis.set_major_formatter(FormatStrFormatter(secymajor_form))
                
              #Interval between minor ticks
             if secyminor_int is not None:newaxvert.yaxis.set_minor_locator(MultipleLocator(secyminor_int))
    
              #Ticks labels color
             if secylab_col is not None:[i_col.set_color(secylab_col) for i_col in newaxvert.get_yticklabels()]
    
         else:
             newaxvert.set_yticks([])

    #-------------------------------------------
      #Hide all axis
    if hide_axis==True:
          ax.xaxis.set_visible(False)
          ax.yaxis.set_visible(False)
          ax.spines['bottom'].set_color('white')
          ax.spines['top'].set_color('white')
          ax.spines['left'].set_color('white')
          ax.spines['right'].set_color('white')
    
              
    return None
    

