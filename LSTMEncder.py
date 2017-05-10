import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
from scipy import signal
from dataset_generate import genDataset,dataReshape
import itertools
from sinegaussian import sinegauss

def andrewNG_model(A,fMin,fMax,sigmaMin,sigmaMax,s_len,win_len,look_back,row_num):

    '''
    Input parameters:

    A (float): Amplitude of sine gaussian signals to be trained on
    fMin (float): Minimum frequency of corresponding sine of signal in training space
    fMax (float): Maximum frequency of corresponding sine of signal in training space
    sigmaMin/sigmaMax: Minimum/Maximum standard deviation of corresponding exponential
                        of signal in training space
    (Refer to sinegaussian.py for further details on the parameters above)

    s_len: length of signal taken (baiscally number of points of sine gaussian signal)
    win_len: length of independent windows (for now disjoint subsets) of the signal taken
    look_back: sliding window on the disjoint bigger windows mentioned above
                Basically the context taken for each point in the input of the neural network
    row_num: loss value outputted in csv file on basis of row number of parameters taken from 
                parameters csv file

    Output:
    avg_loss: Loss computed on the test data used
    more on testing data given below

    Plots also generated for each test signal used.
    '''

    trainX,trainY,trainTarget,trainInput,= genDataset(A,fMin,fMax,sigmaMin,sigmaMax,s_len,win_len,look_back)
    batch_size = 10
    model = Sequential() 
    for i in range(2):
        model.add(LSTM(32, batch_input_shape=(batch_size, win_len , look_back), 
            stateful=True, return_sequences=True))
        model.add(Dropout(0.3))
    model.add(LSTM(32, stateful=True))
    model.add(Dropout(0.3))
    model.add(Dense(win_len))
    model.compile(loss='mean_squared_error', optimizer='adam')
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(trainX, trainY, epochs= 2, batch_size=batch_size, verbose=2,callbacks=[tbCallBack])


    '''
    CUSTOMISED TEST DATA USED HERE (parameters A_, sigmaTest, fTest)
    sigmaTest list of 5 sigmas randomly chosen uniformly from interval (sigmaMin, sigmaMax)
    fTest list of 5 freqs randomly chosen uniformly from interval (fMin, fMax) 
    
    If a signal space is needed for more variety in testing, it is recommended to 
    rewrite the function to have the test space parameters in the function. 
    '''

    A_=[2.0]
    sigmaTest=np.random.uniform(sigmaMin,sigmaMax,5)
    fTest=np.random.uniform(fMin,fMax,5)
    loss=0
    testCount=0
    
    for x in itertools.product(A_,fTest,sigmaTest):
        series=sinegauss(x[0], x[1], x[2], 1000)
        testX,testY=dataReshape(series,series,win_len,look_back)
        Y= model.predict(testX, batch_size = batch_size)
        testCount+=1
        loss += model.evaluate(testX, testY, batch_size = batch_size)
        
        # Following lines are for plotting purpose
        Y = np.reshape(Y, [Y.shape[0]*Y.shape[1],])
        plt.plot(np.arange(len(Y)),Y,'r',label='denoised')
        plt.plot(np.arange(len(Y)),series,'g',label='pure')
        plt.legend()
        plt.savefig("row_{}_Count_{}.jpg".format(row_num,testCount))
        plt.close()
    
    avg_loss = loss/testCount
    return avg_loss



output_values = []

# CSV READ 1
with open('parameters.csv', 'rb') as csvfile:
    param_read = csv.reader(csvfile, delimiter=',', quotechar='|')

    row_num = 0
    for row in param_read:
        row_num += 1
        A = float(row[0])
        fMin = float(row[1])
        fMax = float(row[2])
        sigmaMin = float(row[3])
        sigmaMax = float(row[4])
        s_len = int(row[5])
        win_len = int(row[6])
        look_back = int(row[7])
        param_loss = andrewNG_model(A,fMin,fMax,sigmaMin,sigmaMax,s_len,win_len,look_back,row_num)
        output_values.append(['Row Number'+str(row_num), str(param_loss)])

# CSV READ 2
with open('parameters_loss1.csv', 'wb') as csvfile:
    loss_writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in output_values:
        loss_writer.writerow(row)








