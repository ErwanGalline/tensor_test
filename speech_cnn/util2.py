import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

from scipy.io import wavfile


def graph_spectrogram(wav_file, wav_folder):
    name_save = wav_file.replace(".wav", ".png")
    name_save_cv2 = wav_file.replace(".wav", "_cv2.png")
    rate, data = get_wav_info(wav_file)
    nfft = 256  # Length of the windowing segments
    fs = 256  # Sampling frequency
    plt.clf()
    pxx, freqs, bins, im = plt.specgram(data, nfft, fs)
    plt.axis('off')
    plt.gray()

    plt.savefig(name_save,
                dpi=50,  # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0)

    # Expore plote as image
    fig = plt.gcf()
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    # buf = np.roll(buf, 2)
    cv2.imwrite(name_save_cv2, buf)



def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data


def browse_folder(folder_name):
    for filename in os.listdir(folder_name):
        if filename.endswith(".wav"):
            graph_spectrogram(folder_name + filename, wav_folder)


if __name__ == '__main__':  # Main function
    wav_folder = '/home/erwang/Desktop/mfcc_test/test_pyplot/'
    browse_folder(wav_folder)
