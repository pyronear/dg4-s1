import numpy as np
import matplotlib.pyplot as plt
import io 
import torch

def normalize(v):
    '''Apply z-score normalization to a vector (array)

    Args:
        v (np.array): 1d array

    Returns:
        np.array: normalized array
    '''
    s = np.std(v)
    nv = v-np.mean(v)
    return nv/s

def normalize_2d(v):
    '''Apply z-score normalization to a list of vectors

    Args:
        v (np.array): 2d array

    Returns:
        np.array: normalized array (row wise)
    '''
    s = np.std(v, axis=1).reshape((-1,1))
    m = np.mean(v, axis=1).reshape((-1,1))
    v = np.subtract(v,m)
    return v/s

def smooth(array, n):
    '''Smooth an array by applying moving average

    Args:
        array (np.array): 1d array to smooth
        n (int): sliding window size

    Returns:
        np.array: smoothed array
    '''
    if n%2==0:
        k1 = int(n/2)
        k2 = k1-1
    else:
        k1 = int((n-1)/2)
        k2 = k1
    # extend array as it was a periodic signal
    extended_array = np.concatenate((array[-k1:], array, array[:k2]))
    # apply moving average
    smoothed = np.convolve(extended_array, np.ones(n)/n, mode='valid')
    return smoothed

def add_noise(s, scale=False):
    '''Add moderated gaussian noise to a signal

    Args:
        s (np.array): 1d signal
        scale (bool, optional): Whether or not to add a random scaling factor. Defaults to False.

    Returns:
        np.array: noisy array
    '''
    if scale:
        s = np.add(s, np.random.rand()*1000)
    noise = np.random.normal(np.mean(s)/30, np.std(s)/30, len(s))
    return np.add(noise, s)

def square_diff(v1, v2):
    '''Compute square diff correlation between two signals

    Args:
        v1 (np.array): reference signal
        v2 (np.array): sub signal

    Returns:
        np.array: array of same length as v1, containing a correlation score for each superposition of v2 on v1
    '''
    result = []
    fov = len(v1)
    v1_extended = np.append(v1, v1[:fov+1])
    for i in range(len(v1)):
        v1_window = v1_extended[i:i+len(v2)]
        corr = np.sum(np.square(v1_window-v2))
        result.append(corr)
    return np.array(result)

def square_diff_2d(v1, v2):
    '''Compute square diff correlation between two list of signals

    Args:
        v1 (np.array): list of reference signals
        v2 (np.array): list of sub signals

    Returns:
        np.array (2d): square diff for each ref signal with corresponding sub signal
    '''
    result = []
    fov = len(v2)
    v1_extended = np.append(v1, v1[:,:fov+1], axis=1)
    for i in range(len(v1[0])):
        v1_window = v1_extended[:, i:i+len(v2[0])]
        corr = np.sum(np.square(v1_window-v2), axis=1)
        result.append(corr)
    return np.array(result).T

def square_diff_2d_torch(ref_signals, sub_signals):
    '''Compute square diff correlation between two list of signals (2d tensors), adapted to pytorch

    Args:
        ref_signals (torch.tensor): list of reference signals
        sub_signals (torch.tensor): list of sub signals

    Returns:
        torch.tensor: quare diff for each ref signal with corresponding sub signal
    '''
    # ref_signals has shape (n,l) with n: number of signals, l: length of each signal
    # sub_signals has shape (n,k) with k<l
    n, l = ref_signals.shape 
    k = sub_signals.shape[1]
    # replicate rs and ss along a new dimension
    rs = ref_signals.repeat(l,1,1) # rs has shape (l,n,l)
    ss = sub_signals.repeat(l,1,1) # ss has shape (l,n,k)
    # prepare the windows: indexes of a sliding windows of width k, to scan the ref_signals
    windows = torch.arange(l).reshape(-1,1) # windows has shape (l,1)
    windows = torch.cat(tuple((windows+i)%l for i in range(k)), 1) # windows has shape (l,k)
    windows = windows.repeat(n,1,1).transpose(0,1) # windows has shape (l,n,k) (same as ss)
    # Get all the slices of reference signals with the indexes of windows.
    slices = torch.gather(rs, 2, windows) # slices has shape (l,n,k)
    # compute square diff correlation score along last dimension 
    correlation = torch.sum((slices-ss)**2, 2) # correlation has shape (l,n)
    # transpose to simplify understanding and future usage
    correlation = correlation.transpose(0,1) # correlation has shape (n,l)
    return correlation

def get_ref_signal(skyline, normalization=True):
    '''Ensure the ref signal is of correct length, and normalize it if requested

    Args:
        skyline (np.array): skyline
        normalization (bool, optional): if True, normalize the signal. Defaults to True.

    Returns:
        np.array: reference signal
    '''
    if len(skyline) > 359:
        skyline = skyline[:359]
    ref_signal = normalize(skyline) if normalization else skyline
    return ref_signal

def get_sub_signal(ref_signal, start, fov, normalization=True):
    '''Get a sub signal by slicing a reference signal

    Args:
        ref_signal (np.array): reference signal
        start (int): at which index of the ref signal to start the sub signal
        fov (int): length of the sub signal
        normalization (bool, optional): if True, normalize the signal. Defaults to True.

    Returns:
        np.array: sub signal: slice of ref_signal of length foc
    '''
    sub_signal = np.append(ref_signal, ref_signal[:fov+1])
    sub_signal = sub_signal[start:start+fov]
    sub_signal = normalize(sub_signal) if normalization else sub_signal
    return sub_signal

def get_best_azimuth(corr, fov, argfunc):
    '''Convert correlation scores (array for which the argmin or argmax is the best match) to the azimuth correponding to the center of the view field.
    Concretely, adds half the fov to the best score index.

    Args:
        corr (np.array): correlation scores
        fov (int): field of view
        argfunc (function): argmin or argmax

    Returns:
        int: best azimuth
    '''
    azimuth = (argfunc(corr)+fov/2)%360
    return azimuth

def skylines_to_azimuth(ref_skyline, img_skyline, mode='square-diff'):
    ''' compute sliding square difference (or cross-correlation) between reference signal (panoramic skyline) and shifted/noisy signal (skyline from image)

    Args:
        ref_skyline (np.array): reference signal
        img_skyline (np.array): sub signal
        mode (str, optional): correlation method name. Defaults to 'square-diff'.

    Returns:
        int: best azimuth
    '''
    assert mode in ['square-diff', 'cross-correlation'], 'invalid mode'
    ref_signal = get_ref_signal(ref_skyline)
    img_signal = normalize(img_skyline)
    if mode=='square-diff':
        sqd = square_diff(ref_signal, img_signal)
        azimuth = get_best_azimuth(sqd, len(img_skyline), np.argmin)
    elif mode=='cross-correlation':
        ref_signal = np.append(ref_signal, ref_signal[:2*len(img_signal)])
        cc=np.correlate(ref_signal, img_signal, 'same')
        azimuth = get_best_azimuth(cc, len(img_skyline), np.argmax)
    return azimuth

def plot_skyline(skyline, title):
    '''Plot the maximum elevation angle according to the azimuth angle

    Args:
        skyline (np.array (360,)): the skyline data (max theta for each phi)
        title (str): the plot title
    '''
    plt.style.use('default')
    plt.figure(figsize=(20,5))
    plt.plot(skyline, linewidth=3, color='sienna')
    font_params = {'size':15}
    plt.title(title, fontdict=font_params)
    plt.xlabel("Azimuth angle (°)", fontdict=font_params)
    plt.ylabel("Max elevation angle (°)", fontdict=font_params)
    plt.show()

def plot_skylines_comparison(skyline, img_skyline, plot=True):
    '''Plot two skylines to compare on the same graph

    Args:
        skyline (np.array): reference signal
        img_skyline (np.array): sub signal
        plot (bool, optional): if true, display the plot, in any case returns the plot. Defaults to True.

    Returns:
        np.array: 2d array corresponding to plot data
    '''
    if isinstance(skyline, torch.Tensor):
        s_ref = skyline.detach().numpy() #normalize(skyline)
        s_img = img_skyline.detach().numpy() #normalize(img_skyline)
    else:
        s_ref = skyline
        s_img = img_skyline
    azimuth = np.argmin(square_diff(s_ref, s_img))

    plt.style.use('default')
    fig = plt.figure(figsize=(10,5))
    plt.plot(s_ref, linewidth=3, color='sienna', label='digital terrain model skyline')
    x_img = np.arange(azimuth, azimuth+len(s_img))%359
    # split the arrays if the x goes over 358
    if 0 in x_img and 358 in x_img:
        x_indexes = np.argwhere(x_img==0).flatten()
        x_img = np.split(x_img, x_indexes)
        s_img = np.split(s_img, x_indexes)
    else:
        x_img = [x_img]
        s_img = [s_img]
    for i in range(len(x_img)):
        plt.plot(x_img[i], s_img[i], linewidth=2, color='blue', label='image skyline')
    plt.legend()
    plt.xlabel("Azimuth angle (°)")
    plt.title('Skylines comparison')
    # save the plot in np array
    buf = io.BytesIO()
    plt.savefig(buf, format="raw")
    buf.seek(0)
    img_arr = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8),
                     newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    buf.close()
    # plot only if requested
    if plot:
        plt.show()
    plt.clf()
    plt.close(fig)
    return img_arr