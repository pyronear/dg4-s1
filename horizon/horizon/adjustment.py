import os
import numpy as np
import random
import torch
import torch.nn as nn
from horizon import signal
import matplotlib.pyplot as plt

class SignalLoader():
    def __init__(self):
        self.S0 = []
        self.S1 = []
        self.X = []
        self.Y = []
        self.Y_encode = []
        self.x_train = self.x_val = self.x_test = None
        self.y_train = self.y_val = self.y_test = None

    @staticmethod
    def get_features(array):
        ''' Extract features from a signal
        Args:
            array (np.array): signal to analyze
        Returns:
            list: std, mean, min and max of the signal
        '''
        s = np.std(array)
        m = np.mean(array)
        mini = np.min(array)
        maxi = np.max(array)
        return [s, m, mini, maxi]
    
    @staticmethod
    def get_features_from_pair(ref_signal, sub_signal):
        ''' Extract features from a pair of signals
        Args:
            ref_signal (np.array): reference signal
            sub_signal (np.array): sub signal
        Returns:
            torch.tensor: features ready to give as input to the adjustment model
        '''
        f0 = SignalLoader.get_features(ref_signal)
        f1 = SignalLoader.get_features(sub_signal)
        f = torch.tensor(f0+f1, dtype=torch.float32)
        return f
    
    @staticmethod
    def one_hot_encoding(array, nb_class):
        '''One hot encoding

        Args:
            array (np.array): list of values to encode
            nb_class (int): total number of classes

        Returns:
            np.array (2d): list of encodings, each one of length nb_class
            containing mostly zeros, except 1 at the corresponding value
        '''
        encode = np.zeros((len(array),nb_class))
        value = 1
        np.put_along_axis(encode, array.reshape(-1, 1), value, axis=1)
        return encode
    
    @staticmethod
    def custom_encoding(array, nb_class, margin):
        '''One hot encoding improved to consider surrounding values

        Args:
            array (np.array): list of values to encode
            nb_class (int): total number of classes
            margin (int): values more or less margin to consider

        Returns:
            np.array (2d): list of encodings, each one of length nb_class
            containing mostly zeros, except 1 at the corresponding value and less than 1 (sqrt(x)/x) at the surrounding values
        '''
        encode = np.zeros((len(array),nb_class))
        values = np.abs(np.arange(-margin, margin+1))+1
        values = np.around(np.divide(np.sqrt(values), values), 2)
        indices = np.array([np.arange(i-margin,i+margin+1)%nb_class for i in array])
        np.put_along_axis(encode, indices, values, axis=1)
        return encode
    
    def load_from_dir(self, datadir, fov):
        '''Load data (skylines) from a given directory

        Args:
            datadir (str): path of the directory
            fov (int): field of view, or length of the sub signals
        '''
        # index counter
        i=0
        # takes each signal from a given directory
        for file in os.listdir(datadir):
            # read file
            horizon = np.load(datadir+file)
            ref_signal = signal.get_ref_signal(horizon, normalization=False)
            # smooth signal
            ref_signal = signal.smooth(ref_signal, 5)
            # important to get features before normalization
            f0 = SignalLoader.get_features(ref_signal)
            # extract 4 random subsample for a single skyline
            for _ in range(4):
                # get a random azimuth
                y = random.randint(0,len(ref_signal)-1)
                # slice the reference signal to get the sub signal
                sub_signal = signal.get_sub_signal(ref_signal, y, fov=fov, normalization=False)
                # add random noise and scale
                #sub_signal = signal.add_noise(sub_signal, scale=True)
                # get features
                f1 = SignalLoader.get_features(sub_signal)
                # store data
                self.S0.append(ref_signal)
                self.S1.append(sub_signal)
                self.X.append(f0+f1)
                self.Y.append([i, y])
                i+=1

        # normalize signals (z-score)
        self.S0 = signal.normalize_2d(np.asarray(self.S0))
        self.S1 = signal.normalize_2d(np.asarray(self.S1))
        # adapt format
        self.X = np.asarray(self.X)
        self.Y = np.asarray(self.Y, dtype=int)
        # store encoding for each azimuth
        self.Y_encode = SignalLoader.custom_encoding(self.Y[:,1], 359, 5)
        # convert to torch tensors
        self.S0 = torch.tensor(self.S0, dtype=torch.float32, requires_grad=True)
        self.S1 = torch.tensor(self.S1, dtype=torch.float32, requires_grad=True)
        self.Y_encode = torch.tensor(self.Y_encode, dtype=torch.float32, requires_grad=True)

    @staticmethod
    def split_train_test_val(array, p=0.7):
        '''Split a dataset into 3 (train, test, val), of desired length

        Args:
            array (np.array): dataset to split 
            p (float, optional): proportion of the training set, the test and val sets proportions will be (1-p)/2. Defaults to 0.7.

        Returns:
            np.array (2d): list of each sub array
        '''
        l = len(array)
        n0 = round(l*p)
        n1 = n0+round(l*((1-p)/2))
        return np.split(array, [n0,n1], axis=0)
    
    def prepare_train_test_val(self):
        '''prepare a train, test and validation sets from the loaded data
        '''
        x_train, x_val, x_test = SignalLoader.split_train_test_val(self.X)
        y_train, y_val, y_test = SignalLoader.split_train_test_val(self.Y)

        # convert to torch tensors
        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.x_val = torch.tensor(x_val, dtype=torch.float32)
        self.x_test = torch.tensor(x_test, dtype=torch.float32)

        self.y_train = torch.tensor(y_train, dtype=torch.int)
        self.y_val = torch.tensor(y_val, dtype=torch.int)
        self.y_test = torch.tensor(y_test, dtype=torch.int)

    ## Define losses and evaluation functions
        
    def adjust_signal_batch(self, prediction, target):
        '''Prepare ref and sub signals, adjust sub signals with predictions

        Args:
            prediction (torch.tensor (n,2)): model outputs
            target (torch.tensor (n,2)): Y sample to consider (indexes and azimuths)

        Returns:
            torch.tensor, torch.tensor: reference signals, adjusted sub signals
        '''
        # prediction first column: sub signals weights adjustements
        # prediction first column: sub signals weights adjustements
        # target first column: ref signals indexes
        # target second column: ground truth azimuths
        # retrieve corresponding signals (already normalized)
        ref_signals = self.S0[target[:,0]]
        sub_signals = self.S1[target[:,0]]
        # adjust with predictions
        w = prediction[:,0].reshape(-1,1)
        b = prediction[:,1].reshape(-1,1)
        sub_signals_adjusted = sub_signals * w + b
        return ref_signals, sub_signals_adjusted
    
    @staticmethod
    def adjust_signal(prediction, sub_signal):
        '''Adjust a single signal with prediction

        Args:
            prediction (torch.tensor (2,)): model outputs
            sub_signal (np.array): signal to adjust

        Returns:
            np.array: adjusted signal
        '''
        w = prediction[0].item()
        b = prediction[1].item()
        adjusted_signal = signal.normalize(sub_signal) * w + b
        return adjusted_signal

    def signal_correction_loss(self, prediction, target):
        '''MSE of slices of original signal compared to adjusted sub signals

        Args:
            prediction (torch.tensor (n,2)): model outputs
            target (torch.tensor (n,2)): Y sample to consider (indexes and azimuths)

        Returns:
            float: loss value (lower is better)
        '''
        # get signals tensors
        ref_signals, sub_signals_adjusted = self.adjust_signal_batch(prediction, target)
        # get ground truth azimuth
        azimuths = target[:,1].reshape(-1,1)
        azimuths = torch.cat(tuple((azimuths+i)%359 for i in range(self.S1.shape[1])), 1)
        azimuths = azimuths.long()
        # slice each ref signal within the corresponding azimuth range
        # thus, a slice of the normalized ref signal is obtained
        sub_signals_target = torch.gather(ref_signals, 1, azimuths)
        # and can be compared (MSE) with the adjusted normalized sub signal
        mse_per_row = torch.mean((sub_signals_adjusted-sub_signals_target)**2, 1)
        loss = torch.mean(mse_per_row)
        return loss

    def signals_correlation_loss(self, prediction, target):
        '''More precise loss, but longer to compute
        Negative log likelihood between square diff correlation applied to the adjusted signal compared to the ref signal, and encoded ground truth values

        Args:
            prediction (torch.tensor (n,2)): model outputs
            target (torch.tensor (n,2)): Y sample to consider (indexes and azimuths)
        Returns:
            float: loss value (lower is better)
        '''
        # get signals tensors
        ref_signals, sub_signals_adjusted = self.adjust_signal_batch(prediction, target)
        # compute correlation score
        correlations = signal.square_diff_2d_torch(ref_signals, sub_signals_adjusted)
        # transpose to simplify understanding and future usage
        scores = nn.functional.softmin(correlations, 1)
        # add small amount to avoid to have only 0 leading to an inf loss
        scores = scores +1e-5
        # retrieve target values
        encoded_target = self.Y_encode[target[:,0]]
        # Negative log likelihood
        loss = torch.mean(-1*torch.log(torch.sum(scores*encoded_target, 1))) # scores[:,:-1]
        return loss
    
    # The following functions are not differentiable due to argmin, thus cannot be used as a loss
    # Useful to evaluate
    def azimuth_errors(self, prediction, target):
        '''Compute the difference between predicted azimuth (argmin of square diff) and ground truth azimuth

        Args:
            prediction (torch.tensor (n,2)): model outputs
            target (torch.tensor (n,2)): Y sample to consider (indexes and azimuths)
        Returns:
            torch.tensor: errors for each sub signal
        '''
        # get signals tensors
        ref_signals, sub_signals_adjusted = self.adjust_signal_batch(prediction, target)
        # compute correlation score
        correlations = signal.square_diff_2d_torch(ref_signals, sub_signals_adjusted)
        predicted_azimuths = torch.argmin(correlations, 1).float() # shape (n)
        errors = predicted_azimuths - target[:,1]
        return errors
    
    def mean_azimuth_error(self, prediction, target):
        '''Compute the mean absolute error between predicted azimuth and ground truth azimuth

        Args:
            prediction (torch.tensor (n,2)): model outputs
            target (torch.tensor (n,2)): Y sample to consider (indexes and azimuths)
        Returns:
            float: mean error value (lower is better)
        '''
        errors = self.azimuth_errors(prediction, target)
        mae = torch.mean(torch.abs(errors))
        return mae

    def median_azimuth_error(self, prediction, target):
        '''Compute the median absolute error between predicted azimuth and ground truth azimuth

        Args:
            prediction (torch.tensor (n,2)): model outputs
            target (torch.tensor (n,2)): Y sample to consider (indexes and azimuths)
        Returns:
            float: median error value (lower is better)
        '''
        errors = self.azimuth_errors(prediction, target)
        mae = torch.median(torch.abs(errors))
        return mae

class AdjustmentModel(torch.nn.Module):

    def __init__(self, inputsize=8, load_from=''):
        super(AdjustmentModel, self).__init__()
        self.activation = nn.Tanh()
        self.linear1 = nn.Linear(inputsize, 12)
        self.linear2 = nn.Linear(12, 16)
        self.linear3 = nn.Linear(16, 12)
        self.linear4 = nn.Linear(12, 8)
        self.linear5 = nn.Linear(8, 2)
        self.dropout = nn.Dropout(p=0.3)
        self.train_loss = []
        self.val_loss = []
        self.total_epochs=0
        # load previously saved parameters
        if load_from != '':
            self.load_state_dict(torch.load(load_from))


    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = self.activation(x)
        x = self.linear5(x)
        x = self.activation(x)
        x = x*10
        return x
    
    def train_with_hp(self, loader:SignalLoader, optimizer, loss_fct, epochs, batch_size = 128):
        '''Train the model with given hyperparameters

        Args:
            loader (SignalLoader): data loader
            optimizer (torch.optim): optimizer 
            loss_fct (function): loss function
            epochs (int): number of epochs to train on
            batch_size (int, optional): batch size. Defaults to 128.
        '''
        self.total_epochs += epochs
        self.train() # set train mode
        for epoch in range(epochs):
            for i in range(0, len(loader.x_train), batch_size):
                Xbatch = loader.x_train[i:i+batch_size]
                y_pred = self.forward(Xbatch)
                ybatch = loader.y_train[i:i+batch_size]
                loss = loss_fct(y_pred, ybatch) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.train_loss.append(loss.item())
            self.eval()
            current_val_loss = loss_fct(self.forward(loader.x_val), loader.y_val)
            self.train()
            self.val_loss.append(current_val_loss.item())
            print(f'Epoch {epoch}, train loss {loss.item()}, val loss {current_val_loss.item()}')

    def plot_losses(self, start=0):
        '''Plot the evolution of train and val losses during training

        Args:
            start (int, optional): At which epoch to start the plot. Defaults to 0.
        '''
        plt.style.use('default')
        epochs_x = np.arange(self.total_epochs)
        plt.plot(epochs_x[start:], self.train_loss[start:], color='red', label='train loss')
        plt.plot(epochs_x[start:], self.val_loss[start:], color='orange', label='val loss')
        plt.legend()
        plt.xlabel('epochs')
        plt.xlim([start, epochs_x[-1]])
        plt.plot()

    def save_parameters(self, path):
        '''Save the trained model parameters

        Args:
            path (str): path where to store the model
        '''
        torch.save(self.state_dict(), path)

    def eval_compare(self, loader:SignalLoader):
        '''Evaluate the model on test set, and compare with not adjustments
        Args:
            loader (SignalLoader): data loader
        '''
        self.eval() # set eval mode
        y_pred = self.forward(loader.x_test)
        print('Test correction loss', loader.signal_correction_loss(y_pred, loader.y_test).item())
        print('Test correlation loss', loader.signals_correlation_loss(y_pred, loader.y_test).item())
        # signal multiplied by 1 and added to 0 is equivalent to no change
        no_change = torch.empty_like(loader.y_test)
        no_change[:,0] = 1 
        no_change[:,1] = 0
        # compute the mean azimuth error without adjusting the sub signals
        err_no_change = loader.mean_azimuth_error(no_change, loader.y_test).item()
        med_no_change = loader.median_azimuth_error(no_change, loader.y_test).item()
        #  compute the mean azimuth error of azimuth estimation with the neural network estimated adjustements
        err_pred = loader.mean_azimuth_error(y_pred, loader.y_test).item()
        med_pred = loader.median_azimuth_error(y_pred, loader.y_test).item()
        print(f'Mean error without adjustement: {err_no_change}, Median error: {med_no_change}')
        print(f'Mean error with adjustement: {err_pred}, Median error: {med_pred}')
        # boxplot 
        errors_no_change = loader.azimuth_errors(no_change, loader.y_test).detach().numpy()
        errors_pred = loader.azimuth_errors(y_pred, loader.y_test).detach().numpy()
        flier_props = {'marker': '.', 'markersize': 2}
        plt.boxplot([errors_no_change, errors_pred], labels=['without adjustment', 'with adjustment'], flierprops=flier_props)
        plt.ylabel('azimuth error')
        plt.title('Distribution of azimuth errors')
        plt.show()