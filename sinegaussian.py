import numpy as np
import matplotlib.pyplot as plt

def sinegauss(A, fc, sigma, n, l_lt=-1, r_lt=1):
	
	'''
	Input parameters:

	A  (float): represents the amplitude of the corresponding sine wave
	fc (float): represents the frequency of the corresponding sine wave
	sigma (float): standard deviation of the corresponding decaying exponential
	n (int): number of points in the signal
	l_lt (float): left limit of signal on X axis (where signal becomes non-zero)
	r_lt (float): right limit of signal on X axis (where signal becomes zero again)

	Output:
	sg: List consisting of n points (floats) on the sine gaussian signal intended 
	'''

	tau=np.random.uniform(l_lt,r_lt)
	t = np.linspace(l_lt, r_lt, n)
	sg = A * np.sin(2*np.pi*fc*t) * np.exp(-0.5*(t-tau)**2./sigma**2.)
	return sg