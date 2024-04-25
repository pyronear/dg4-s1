import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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

def pad_smooth(array, n=5):
    '''Smooth an array with padding first to ensure same length.

    Args:
        array (np.array): signal to smooth
        n (int): smoothing factor

    Returns:
        np.array: smoothed array of same length
    '''
    padded = np.pad(array, (n//2, n-1-n//2), mode='edge')
    smoothed = np.convolve(padded, np.ones(n)/n, mode='valid')
    return smoothed

def smooth(array, n):
    '''Smooth an array by applying moving average. Made for periodic signals (end can be continued with beginning)

    Args:
        array (np.array): 1d array to smooth
        n (int): sliding window size (n>1)

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

def smooth_with_depth(array, depths):
    '''Smooth an array by applying moving average.
    The window size depends on the depth of each point

    Args:
        array (np.array): 1d array to smooth
        array (np.array): sliding window sizes

    Returns:
        np.array: smoothed array
    '''
    def get_window_size(n):
        if n%2==0:
            k1 = int(n/2)
            k2 = k1-1
        else:
            k1 = int((n-1)/2)
            k2 = k1
        return k1, k2
    
    k0, kx = get_window_size(depths.max())
    # first smooth depths to avoid alternating between far and close (can happen if too sparse points)
    depths = np.round(smooth(depths, 5))
    # extend array as it was a periodic signal
    extended_array = np.concatenate((array[-k0:], array, array[:kx]))
    # apply moving average with changing window size
    smoothed = np.empty_like(array)
    for i, d in enumerate(depths):
        k = i+k0
        k1, k2 = get_window_size(d)
        smoothed[i] = np.mean(extended_array[k-k1:k+k2+1])
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

def ternary_slope(s, n=2):
    '''Extract the slopes of a signal. Cast to 0 (flat), 1 (up), or -1 (down)

    Args:
        v (np.array): array to "derive"
        n (int, optional): slope observation range. Defaults to 2.

    Returns:
        np.array: slope values
    '''
    result = np.empty_like(s)
    iqr = np.subtract(*np.percentile(s, [75, 25]))
    a = iqr/100
    for i in range(len(s)-n):
        if s[i]-a<=s[i+n]<=s[i]+a:
            result[i] = 0
        else:
            diff = s[i]>s[i+n]
            result[i] = -1 if diff else 1
    return result[:-1]

def slope_square_diff(v1, v2, n=2):
    '''Compute square diff correlation from slopes. Better than simple square_diff()

    Args:
        v1 (np.array): reference signal
        v2 (np.array): sub signal
        n (int, optional): slope observation range. Defaults to 2.

    Returns:
        np.array: array of same length as v1, containing a correlation score for each superposition of v2 on v1
    '''
    result = []
    fov = len(v2)
    v1_extended = np.append(v1, v1[:fov+1])
    v1_extended = ternary_slope(v1_extended, n)
    v2 = ternary_slope(v2, n)
    for i in range(len(v1)):
        v1_window = v1_extended[i:i+fov-1]
        corr = np.sum(np.square(v1_window-v2))
        result.append(corr)
    return np.array(result)

def square_diff(v1, v2, with_offset=False):
    '''Compute square diff correlation between two signals

    Args:
        v1 (np.array): reference signal
        v2 (np.array): sub signal
        with_offset (bool, optional): if true, try to correct y-offset before computing score

    Returns:
        np.array: array of same length as v1, containing a correlation score for each superposition of v2 on v1
    '''
    result = []
    fov = len(v2)
    v1_extended = np.append(v1, v1[:fov+1])
    for i in range(len(v1)):
        v1_window = v1_extended[i:i+fov]
        offset = np.mean(v2-v1_window) if with_offset else 0
        corr = np.sum(np.square(v1_window-v2-offset))
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
    if len(skyline) > 360:
        skyline = skyline[:360]
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

def skylines_to_azimuth(ref_skyline, img_skyline, mode='slope-square-diff', center=True):
    ''' Compute sliding correlation score between reference signal (panoramic skyline) and shifted/noisy signal (skyline from image). Gives the best match

    Args:
        ref_skyline (np.array): reference signal
        img_skyline (np.array): sub signal
        mode (str, optional): correlation method name. Defaults to 'slope-square-diff'.
        center (bool, optional): If True (default), adds an offset of fov/2 to the best correlation match. 
        In other words, returns the azimuth corresponding to the center of the image (whereas center=False gives the leftmost azimuth of the signal)

    Returns:
        int: best azimuth
    '''
    assert mode in ['square-diff','slope-square-diff', 'cross-correlation'], 'invalid mode'
    ref_signal = normalize(ref_skyline) #get_ref_signal()
    img_signal = normalize(img_skyline)
    fov = len(img_skyline) if center else 0
    if mode=='square-diff':
        sqd = square_diff(ref_signal, img_signal)
        azimuth = get_best_azimuth(sqd, fov, np.argmin)
    elif mode=='slope-square-diff':
        sqd = slope_square_diff(ref_signal, img_signal)
        azimuth = get_best_azimuth(sqd, fov, np.argmin)
    elif mode=='cross-correlation':
        ref_signal = np.append(ref_signal, ref_signal[:2*len(img_signal)])
        cc = np.correlate(ref_signal, img_signal, 'same')
        azimuth = get_best_azimuth(cc, fov, np.argmax)
    return azimuth

def plot_skyline(skyline, title, depths=None):
    '''Plot the maximum elevation angle according to the azimuth angle

    Args:
        skyline (np.array (360,)): the skyline data (max theta for each phi)
        title (str): the plot title
        depths (np.array (360,), optional): if specified, color the line according to the depth values
    '''
    plt.style.use('default')
    plt.figure(figsize=(20,5))
    if isinstance(depths, np.ndarray):
        cmap = plt.get_cmap('copper', 8)
        plt.scatter(np.arange(360), skyline, s=40, c=10-depths, cmap=cmap, edgecolor='none')
    else:
        plt.plot(skyline, linewidth=3, color='sienna')
    font_params = {'size':15}
    plt.title(title, fontdict=font_params)
    plt.xlabel("Azimuth angle (°)", fontdict=font_params)
    plt.ylabel("Max elevation angle (°)", fontdict=font_params)
    plt.show()

def plot_skylines_comparison(skyline, img_skyline, force_azimuth=-1, plot=True):
    '''Plot two skylines to compare on the same graph

    Args:
        skyline (np.array): reference signal
        img_skyline (np.array): sub signal
        force_azimuth (int, optional): if filled, force the position of sub signal
        plot (bool, optional): if true, display the plot, in any case returns the plot. Defaults to True.

    Returns:
        np.array: 2d array corresponding to plot data
    '''
    if isinstance(skyline, torch.Tensor):
        s_ref = skyline.detach().numpy()
        s_img = img_skyline.detach().numpy()
    else:
        s_ref = skyline
        s_img = img_skyline
    if force_azimuth >= 0:
        azimuth = force_azimuth
    else:
        azimuth = skylines_to_azimuth(s_ref, s_img, center=False)

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