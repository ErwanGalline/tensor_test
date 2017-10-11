from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import librosa
import cv2


def init():
    directory = "/home/erwang/Desktop/"
    wav = "0e17f595_nohash_0.wav"
    convert_wav(directory, wav)


def convert_wav(dir, wav):
    wave, sr = librosa.load(dir + wav,sr=None,  mono=True)
    mfcc = librosa.feature.mfcc(wave, sr ,n_mfcc=20)
    # mfcc = np.pad(mfcc, ((0, 0), (0, 80 - len(mfcc[0]))), mode='constant', constant_values=0)

    wav = wav.replace(".wav", "")
    cv2.imwrite("/home/erwang/Desktop/" + wav + ".png", np.array(mfcc))


if __name__ == "__main__":
    init()
