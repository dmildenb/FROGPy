### FROG Plotting Function
# Daniel Mildenberger
# Ported on June 23rd, 2016
# Last Updated June 23rd, 2016

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

## Plotting Functions

def FROG_plot(Trace_w, Pulse, Gate, IFROG, G=[], Cen_Wave=0, Wave_PPx=1, Delay_PPx=1):
#     Cen_Wave = 743.6233 # Center Wavelength in nm
#     Wave_PPx = 0.7432871 # Wavelength per pixel in nm
#     Delay_PPx = 1 # Delay time per pixel

    ax1=Trace_plot(np.absolute(IFROG))
    ax1.set_title('Experimental Trace',fontsize=20)

    ax2 = Trace_plot(np.absolute(Trace_w)**2)
    ax2.set_title('Retrieved Trace',fontsize=20)
    
    ax3 = Trace_plot(np.real(IFROG - np.absolute(Trace_w)**2))
    ax3.set_title('Difference (Measured - Retrieved)')

    ax4,ax5 = Field_plot(fft.fftshift(fft.fft(Pulse)), fft.fftshift(fft.fft(Gate)), Freq_Axis_Spect(IFROG.shape[0], Cen_Wave, Wave_PPx))
    ax4.set_title('Spectral Characteristics',fontsize=20)
    ax4.set_xlabel('Frequency (a.u.)')
    
    ax6,ax7 = Field_plot(Pulse, Gate, Delay_Axis(IFROG.shape[0], Delay_PPx))
    ax6.set_title('Temporal Characteristics',fontsize=20)
    ax6.set_xlabel('Time (a.u.)')
    
    if G != []:
        fig,ax=plt.subplots(figsize=(10,8))
        ax.plot(G)
        ax.set_title('FROG Error', fontsize=20)
    
    return ax1, ax2, ax3, ax4, ax5, ax6, ax7
	
# Plots the magnitude and phase of a complex pulse with the given axis.
def Field_plot(Probe, Gate, x):
    fig,ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    
    ax1.plot(x,np.absolute(Probe),label='Probe Mag.')
    ax1.plot(x,np.absolute(Gate),label='Gate Mag.')
    ax2.plot(x,np.unwrap(np.angle(Probe)),label='Probe Phase')
    ax2.plot(x,np.unwrap(np.angle(Gate)),label='Gate Phase')
    ax1.set_ylabel('Magnitude (a.u.)')
    ax2.set_ylabel('Phase (Rad)')
    ax1.legend()

    return ax1,ax2
    
# Plots the magnitude of a complex 2d trace.
def Trace_plot(Trace_w):
    fig,ax=plt.subplots(figsize=(10,8))
    ax1=ax.imshow(Trace_w)
    ax.set_title('Intensity Trace')
    plt.colorbar(ax1)
    ax.set_xlabel('Delay (a.u.)')
    ax.set_ylabel('Wavelength (a.u.)')
    return ax

def Gauss_Axis(Width):
    return np.linspace(-Width/2, Width/2, Width, dtype=complex)

def Freq_Axis_Trace(PixelCount, CenterWavelength, WavelengthPerPixel):
    return np.linspace(CenterWavelength - PixelCount * WavelengthPerPixel / 2, \
                       CenterWavelength + PixelCount * WavelengthPerPixel / 2, \
                       PixelCount)

def Freq_Axis_Spect(PixelCount, CenterWavelength, WavelengthPerPixel):
    return 0.5 * np.linspace(CenterWavelength - PixelCount * WavelengthPerPixel / 2, \
                             CenterWavelength + PixelCount * WavelengthPerPixel / 2, \
                             PixelCount)

def Delay_Axis(PixelCount, DelayPerPixel):
    return np.linspace(-PixelCount * DelayPerPixel / 2, PixelCount * DelayPerPixel, PixelCount)
	
	