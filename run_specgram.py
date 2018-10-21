"""
run_specgram.py

Main Script for the Live Spectrogram project, a real time spectrogram
visualization tool

Dependencies: matplotlib, numpy and the mic_read.py module
"""
############### Import Libraries ###############
import matplotlib
# gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
# for gui in gui_env:
#     try:
#         print("testing", gui)
#         matplotlib.use(gui,warn=False, force=True)
#         from matplotlib import pyplot as plt
#         break
#     except:
#         continue
# print("Using:",matplotlib.get_backend())
import scipy
from scipy import signal

from matplotlib.mlab import window_hanning,specgram
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import numpy as np

############### Import Modules ###############
import mic_read


############### Helper Functions ############
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

############### Constants ###############
SAMPLES_PER_FRAME = 50   #Number of mic reads concatenated within a single window
nfft = 2048 #NFFT value for spectrogram
overlap = 512 #overlap value for spectrogram
rate = mic_read.RATE #sampling rate

lowcut = 500 # Hz # Low cut for our butter bandpass filter
highcut = 15000 # Hz # High cut for our butter bandpass filter

##### Style specific hyperparams
## Noise Cancellation + Normalisation 
eps = 1e-17
thresh = -10
# Beethoven
# eps = 1e-17
# thresh = -5
# YLIMIT = 1000
# cmap = plt.cm.afmhot
YLIMIT = 4500
CMAP = 'viridis'



############### Functions ###############
"""
get_sample:
gets the audio data from the microphone
inputs: audio stream and PyAudio object
outputs: int16 array
"""
def get_sample(stream,pa):
    data = mic_read.get_data(stream,pa)
    data = butter_bandpass_filter(data, lowcut, highcut, rate, order=1)
    return data
"""
get_specgram:
takes the FFT to create a spectrogram of the given audio signal
input: audio signal, sampling rate
output: 2D Spectrogram Array, Frequency Array, Bin Array
see matplotlib.mlab.specgram documentation for help
"""
def get_specgram(signal,rate):
    freqs,bins, arr2D = scipy.signal.spectrogram(
        x=signal,
        fs=rate,
        noverlap=overlap,
        nperseg=nfft,
        nfft=nfft,
        detrend=False,
        return_onesided=True
    )
    # arr2D,freqs,bins = specgram(
    #                             signal,
    #                             window=window_hanning,
    #                             Fs = rate,
    #                             NFFT=nfft,
    #                             detrend="linear",
    #                             noverlap=overlap
    #                             )
    arr2D /= arr2D.max()
    arr2D = np.log(arr2D + eps)
    arr2D[arr2D < thresh ] = thresh
    
    return arr2D,freqs,bins

"""
update_fig:
updates the image, just adds on samples at the start until the maximum size is
reached, at which point it 'scrolls' horizontally by determining how much of the
data needs to stay, shifting it left, and appending the new data. 
inputs: iteration number
outputs: updated image
"""
def update_fig(n):
    data = get_sample(stream,pa)
    arr2D,freqs,bins = get_specgram(data,rate)
    im_data = im.get_array()
    if n < SAMPLES_PER_FRAME:
        im_data = np.hstack((im_data,arr2D))    
        im.set_array(im_data)
    else:
        keep_block = arr2D.shape[1]*(SAMPLES_PER_FRAME - 1)
        im_data = np.delete(im_data,np.s_[:-keep_block],1)
        im_data = np.hstack((im_data,arr2D))
        im.set_array(im_data)
    return im,

############### Initialize Plot ###############
fig = plt.figure()
"""
Launch the stream and the original spectrogram
"""
stream,pa = mic_read.open_mic()
data = get_sample(stream,pa)
arr2D,freqs,bins = get_specgram(data,rate)
print(arr2D.min(), arr2D.max())
"""
Setup the plot paramters
"""
extent = (bins[0],bins[-1]*SAMPLES_PER_FRAME,freqs[-1],freqs[0])

im = plt.imshow(arr2D,aspect='auto',extent = extent,interpolation="bicubic",
                cmap = CMAP)

# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.title('Real Time Spectogram')
plt.gca().invert_yaxis()
# plt.axis('off')
# plt.yscale('log')
plt.ylim((50, YLIMIT))
##plt.colorbar() #enable if you want to display a color bar

############### Animate ###############
anim = animation.FuncAnimation(fig,update_fig,blit = False,
                               interval=mic_read.CHUNK_SIZE/2000)
try:
    plt.show()
except:
    print("Plot Closed")

############### Terminate ###############
stream.stop_stream()
stream.close()
pa.terminate()
print("Program Terminated")
