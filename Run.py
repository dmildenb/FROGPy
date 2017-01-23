### FROG Loop
# Daniel Mildenberger
# Ported on October 4th, 2016
# Last Updated on October 4th, 2016

import numpy as np
import FROG_lib as lib

## Notes

# Overcorrection for ProjI? (Erf profile? Linear?) [This doesn't seem to help much with complex pulses]
# Marginals?
# SHG Intensity Integral Constraint?
# Pulse Centering
# Time, Frequency and Delay Calibration -> Proper Axis Displayed
# Import into LabView?

# A lot of reorganization, rewriting for readability and optimization has to happen with the defined functions.

# Finally, they cheat around the matrix multiplication for finding OOT and OTO by returning the vector-
# matrix multiplication with the guess vector. (They also normalize the peak height of these vectors to one.)

### Frog Loop Function

def FROG_Loop(IFROG, Pulse=[], Gate=[], max_iter=100, G_min=10**-5):
    
    ## Initialization
    current = int(0) # Current iteration
    G = np.zeros(max_iter) # FROG error at each iteration
    G_best = float("inf") # Sets current best FROG error to infinity
    PCGP_Power = 1 # Sets the matrix power to apply in the PCGP algorithm
        
    Best_Trace_w = np.zeros(IFROG.shape, dtype=complex) # Preallocates for best retrieved trace
    Best_Pulse = np.zeros(IFROG.shape[0], dtype=complex) # Preallocates for best retrieved Probe vector
    Best_Gate = np.zeros(IFROG.shape[0], dtype=complex) # Preallocates for best retrieved Gate vector
    
    # Generates complex vectors with Gaussian amplitude and random phase for use as the initial guess
    if Pulse == []:
        Pulse = lib.Gaussian(lib.Gauss_Axis(IFROG.shape[0]), IFROG.shape[0]//3) * np.exp(1j * \
                        (np.random.rand(IFROG.shape[0]) - 0.5) * np.pi)
    if Gate == []:
        Gate = lib.Gaussian(lib.Gauss_Axis(IFROG.shape[0]), IFROG.shape[0]//3) * np.exp(1j * \
                        (np.random.rand(IFROG.shape[0]) - 0.5) * np.pi)
        
    Shift_idx = lib.OP_Shift_Calc(IFROG.shape) # Linear index for shifting to/from outer product form
    Trace_w = lib.FROG_fft(np.take(lib.FieldTrace_SHG2(Pulse, Gate), Shift_idx)) # Generates the initial guess trace
    
    ## FROG Loop

    # Loop runs until max iteration number or target error is reached.
    while (current < max_iter) & (G_best > G_min):
        # Project measured magnitude onto current trace
        Trace_w = lib.ProjI(Trace_w, IFROG)
   
        # Inverse Fourier transform and shift trace to time domain
        Trace_t = lib.FROG_ifft(Trace_w)
   
        # Apply pseudo SVD power method to inverse-shifted trace to recover probe and gate vectors
        Pulse, Gate = lib.PCGP_FROG(np.take(Trace_t, Shift_idx), Pulse, Gate, PCGP_Power)
        
        # Reconstruct and shift the time domain trace from the probe and gate vectors
        Trace_t = np.take(lib.FieldTrace_SHG2(Pulse, Gate), Shift_idx)
    
        # Fourier transform and shift trace to frequency domain
        Trace_w = lib.FROG_fft(Trace_t)
    
        # Calculate FROG error of current trace
        G[current] = lib.Gerrshg(Trace_w, IFROG)
    
        # Keep track of best retrieved error, trace, probe and gate
        if G[current] < G_best:
            G_best = G[current]
            Best_Trace_w = Trace_w[:,:]
            Best_Pulse = Pulse[:]
            Best_Gate = Gate[:]
    
        current += 1
        
    Best_Trace_w /= np.amax(np.absolute(Best_Trace_w))
    
    return Best_Trace_w, Best_Pulse, Best_Gate, G


