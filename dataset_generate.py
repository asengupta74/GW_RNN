import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import itertools
from sinegaussian import sinegauss

def genSeries(A, fc, sigma, n,noise_add):
	'''
	Input parameters:
	A, fc, sigma, n: Refer to sinegaussian documentation
	noise_add: if 1, then gaussian noise with mean 0 and std dev 1 added to signal

	Output:
	List consisting of points of noise/noiseless signal added
	'''
	prepad=np.zeros(n)
	series=sinegauss(A, fc, sigma, n, l_lt=-1, r_lt=1)
	series=np.append(prepad,series)
	postpad=np.zeros(n)
	data=np.append(series,postpad)
	if noise_add:
		noise = np.random.normal(0, 1, len(data))
		data+=noise
	return data

def dataReshape(noisy,pure,win_len,look_back):
	'''
	Input:
	noisy: list consisting of noisy data points to be used as network input
	pure: list consisting of pure data points to be used as network output
	win_len: length of disjoint windows (subsets) of the signals used
	look_back: sliding window on the disjoint bigger windows mentioned above
               Basically the context taken for each point in the input of 
               the neural network

    Output:
    dataX: array of dimension (lenX/win_len, win_len, lookback) used as network input
    		A context of size of lookback goes into each neuron in the input lyr of network
    dataY: array of dimension (lenX/win_len, win_len) used as network output
    		Each context corresponds to one denoised point, contained in dataY array.

	'''
	X=noisy
	Y=pure
	dataX,dataY=[],[]
	num_points=len(X)
	num_windows =  num_points/win_len
	X = np.array_split(X, num_windows)
	X = np.asarray(X)
	Y = np.split(Y, num_windows)
	Y = np.asarray(Y)
	for i in range(len(X)):
		c = X[i]
		d = Y[i]
		win = []
		win_y = []
		slice_index = int(look_back/2)

		'''
		Lines 63- 70 are for zero padding required in starting of the window
		'''
		
		for j in range(0, slice_index):
			#num_zeroes = slice_index - j
			win_y.append(d[j])
			chop1 = c[0:j+slice_index+1]
			num_zeroes = look_back - len(chop1)
			chop = np.zeros(num_zeroes)
			chop=np.append(chop, c[0:j+slice_index+1])
			win.append(chop)
			# print chop

		'''
		Lines 77 - 80 are for zero padding required in end of the window
		'''

		for j in range(slice_index, len(c)-slice_index):
			chop = c[j-slice_index:j+slice_index+1]
			win_y.append(d[j])
			win.append(chop)
			# print chop, len(chop), type(chop)

		for j in range(len(c)-slice_index, len(c)):
			#num_zeros = look_back - (len(c)-1-j-)
			chop = c[j-slice_index:len(c)]
			num_zeros = look_back - len(chop)
			# chop = chop + ([0]*num_zeros)
			# print "HELLO", chop, np.zeros(num_zeros)
			chop=np.append(chop, np.zeros(num_zeros))
			win.append(chop)
			# print chop
			win_y.append(d[j])

		dataY.append(win_y)
		dataX.append(win)
	dataX=np.asarray(dataX)
	dataY=np.asarray(dataY)
	return dataX,dataY

def genDataset(A,fmin,fmax,sigmaMin,sigmaMax,s_len,win_len,look_back):

	'''
	Refer to LSTMEncdr.py for explanation of parameters.
	'''
	
	f=np.linspace(fmin,fmax,fmax-fmin+1)
	A_=[A]
	sigma=np.linspace(sigmaMin,sigmaMax,int(sigmaMax/sigmaMin))
	data=[]
	for p in itertools.product(A_,f,sigma):
		pure=genSeries(p[0], p[1], p[1], s_len,0)
		noisy=genSeries(p[0], p[1], p[1], s_len,1)
		data.append([pure,noisy])
	data=np.asarray(data)
	for i in range(10):
		np.random.shuffle(data)
	#splitIdx=int(data.shape[0]*0.7)
	#train=data[:splitIdx]
	#test=data[splitIdx:]

	train = data
	pt=train[:,0,:]
	nt=train[:,1,:]
	#trainTarget=np.reshape(pt,[pt.shape[0]*pt.shape[1],])
	#trainInput=np.reshape(nt,[nt.shape[0]*nt.shape[1],])
	trainX,trainY=dataReshape(trainInput,trainTarget,win_len,look_back)

	#pt=test[:,0,:]
	#nt=test[:,1,:]
	#testTarget=np.reshape(pt,[pt.shape[0]*pt.shape[1],])
	#testInput=np.reshape(nt,[nt.shape[0]*nt.shape[1],])
	#testX,testY=dataReshape(testTarget,testInput,win_len,look_back)
	
	#testTarget = sinegauss(2.0, 5.0, 0.25, 0.1, s_len, -1, 1)
	# testTarget = sinegauss(A_test, f_test, sig_test, test_tau, s_len, l_lt, r_lt)
	# testInput = testTarget
	# testX, testY = dataReshape(testTarget, testInput, win_len, look_back)
	# print "testX-->",testX.shape
	# print "testY-->",testY.shape
	# print "trainX-->",trainX.shape
	# print "trainY-->",trainY.shape
	# return trainX,trainY,testX,testY,trainTarget,trainInput,testTarget,testInput
	return trainX,trainY,trainTarget,trainInput
