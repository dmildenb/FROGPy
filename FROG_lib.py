### FROG Code Functions
# Daniel Mildenberger
# Ported on June 20th, 2016

import numpy as np
import numpy.linalg as nla
import numpy.fft as fft

## Functions

# Calculates FROG error in accordance with "Frequency-resolved optical gating with the use of
# second-harmonic generation (Trebino 1994)." Magnitudes of the measured and retrieved traces 
# are normalized to 1.
def Gerrshg(Trace_w, IFROG):
    Trace = np.absolute(Trace_w)
    Trace /= np.amax(Trace)
    return np.sqrt(np.sum((IFROG - np.absolute(Trace*Trace))**2)) / Trace_w.shape[0]

# Projects the amplitude of the measured spectra onto the retrieved trace in accordance with 
# "Improved ultrashort pulse-retrieval algorithm for frequency-resolved optical gating
# (Trebino 1994)."
def ProjI(Ew, I_FROG):
    Ew /= np.absolute(Ew)
    Ew *= np.sqrt(I_FROG)
    Ew[np.isnan(Ew)] = 0
    return Ew

# This function call could probably just be replaced in line.
# Fourier transforms the columns of the time domain trace and applies an fft shift.
def FROG_fft(Etsig):
    return fft.fftshift(fft.fft(Etsig, axis=0), axes=0)

# This function call could probably just be replaced in line.
# Inverse Fourier transforms the columns of the time domain trace and applies an ifft shift.
def FROG_ifft(Ewsig):
    return fft.ifft(fft.ifftshift(Ewsig, axes=0), axis=0)

# Applies the "Power Method" pseudo SVD with a preset power of 1 to recover the principal components 
# (probe and gate vectors) of a matrix in outer product form. Performs the same function as 
# PCGP_FROG_Power, but faster and without a vairable power. The returned vectors are normalized a peak of one.
def PCGP_FROG(Out_Prod, Guess_Pulse, Guess_Gate, Power):  
    OOT = np.dot(Out_Prod, np.conj(np.matrix.transpose(Out_Prod)))
    OTO = np.dot(np.conj(np.matrix.transpose(Out_Prod)), Out_Prod)
    Pulse = np.dot(nla.matrix_power(OOT, Power), Guess_Pulse)
    Gate = np.dot(nla.matrix_power(OTO, Power), Guess_Gate)
    Pulse = Pulse / np.amax(np.absolute(Pulse))
    Gate = Gate / np.amax(np.absolute(Gate))
    return Pulse, Gate

# This function call could probably just be replaced in line.
# Calculates the time domain trace in outer product form from the recovered probe and gate vectors.
# See "Recent Progress Toward Real-Time Measurement of Ultrashort Laser Pulses (Kane 1999)."
def FieldTrace_SHG2(Pulse, Gate):
    return np.outer(np.conj(Pulse), Gate) + np.outer(Gate, np.conj(Pulse))

# Shifts elements of a vector to the left by the given amount.
def Vec_shift_L(vec, shift=0):
    s = vec.size
    out = np.zeros(s, dtype=complex)
    out[:s-shift] = vec[shift:]
    out[s-shift:] = vec[:shift]
    return out

# Shifts elements of a vector to the right by the given amount.
def Vec_shift_R(x,shift=0):
    s=x.size
    y=np.zeros(s, dtype=complex)
    y[:shift]=x[s-shift:]
    y[shift:]=x[:s-shift]
    return y

# Shifts a outer product matrix into time domain trace form. Rows are shifted left by one less than
# their row number (if not zero indexing) and then the left and right halves of the matrix are 
# interchanged. Here this is accomplished in one step by shifting left by one less than the row 
# number in addition to a shift left by half the matrix width. For inverse, see iOP_Shift.
# See "Principal Components Generalized Projections: A Review (Kane 2008)."
# def OP_Shift(Trace):
#     s = Trace.shape
#     Out = np.zeros(s, dtype=complex)
#     shifts = -((np.arange(s[0])+s[0]/2)%s[0])
    
#     for i in np.arange(s[0]):
#         Out[i,:] = Vec_shift_L(Trace[i,:],(i+s[0]/2)%s[0])
#         Out[i,:] = np.roll(Trace[i,:],-((i+s[0]/2)%s[0]))
#     Out = np.roll(Trace, shifts, axis=1)

#     for i in np.arange(s[0]):
#         Out[i,:] = np.flipud(Out[i,:])
    
#     return Out

# This could still be improved by calculating the shift matrix once prior to entering the FROG loop and 
# just calling np.take inside the loop.
# def OP_Shift(Trace):
#     s = Trace.shape
    
#     I = np.arange(s[0])
#     J = np.arange(s[1]-1,-1,-1)
    
#     shifts = (I + s[0]/2) % s[0]
    
#     linear_idx = (((shifts[:,None] + J)%s[1]) + I[:,None]*s[1])
    
#     return np.take(Trace, linear_idx)

def OP_Shift_Calc(MatShape):
    I = np.arange(MatShape[0])
    J = np.arange(MatShape[1]-1,-1,-1)
    shifts = (I + MatShape[0]/2) % MatShape[0]
    linear_idx = (((shifts[:,None] + J)%MatShape[1]) + I[:,None]*MatShape[1])
    return linear_idx.astype(int)

# Inverse to OP_Shift. Shifts a matrix from time domain trace form to outer product form. Rows are 
# shifted right by one less than their row number (if not zero indexing) in addition to a shift 
# right by half the matrix width. See OP_Shift.
# def iOP_Shift(Trace):
#     s = Trace.shape
#     Out = np.zeros(s, dtype=complex)
#     for i in np.arange(s[0]):
#         Out[i,:] = np.flipud(Trace[i,:])
        
#     for i in np.arange(s[0]):
#         Out[i,:] = Vec_shift_R(Out[i,:],(i+s[0]/2)%s[0])
#         Out[i,:] = np.roll(Out[i,:],(i+s[0]/2)%s[0])
    
#     return Out

# Returns a Gaussian envelope given an axis sampling and a full-width half-max
def Gaussian(x, FWHM):
    sigma = FWHM / np.sqrt(8 * np.log(2))
    return np.exp(-x**2 / 2 / sigma**2)

# 
def Gauss_Axis(Width):
    return np.linspace(-Width/2, Width/2, Width, dtype=complex)

# Plots the magnitude and phase of a complex pulse with the given axis.
def Field_plot(Pulse, x):
    fig,ax1 = plt.subplots(figsize=(10,8))
    ax1.plot(x,np.absolute(Pulse))
    ax1.set_title('Retrieved Pulse Magnitude')
    
    fig,ax2 = plt.subplots(figsize=(10,8))
    ax2.plot(x,np.unwrap(np.angle(Pulse)))
    ax2.set_title('Retrieved Pulse Phase')

    return ax1,ax2
    
# Plots the magnitude of a complex 2d trace.
def Trace_plot(Trace_w):
    fig,ax=plt.subplots(figsize=(10,8))
    ax1=ax.imshow(Trace_w)
    ax.set_title('Intensity Trace')
    plt.colorbar(ax1)
    return ax

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

