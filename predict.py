import torch.nn as nn  
import torch 
import numpy as np 
from train_tools import *
from torch.utils.data import DataLoader
import os
import librosa
from data_tools import *

sample_rate = 8000
frame_length = 8064
stride_noise = 5000
stride_clean = 8064
stride_fft = 63
min_len = 1.0
num_sample =  50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE2().to(device)
model.load_state_dict(torch.load('../model/best_model5.pth', map_location=device))
model.eval()

demo_audio_path = '../demo/'
noisy_voice = audio_to_numpy(demo_audio_path, sample_rate, frame_length, stride_clean, min_len)

m_amp_db_audio,  m_pha_audio = numpy_to_spec(
        noisy_voice, stride_fft)
X = scaled_in(m_amp_db_audio)
X = X.reshape(X.shape[0],1,X.shape[1],X.shape[2])
X = torch.FloatTensor(X).to(device)
_ , noice_pred = model(X)
inv_sca_noice_pred = inv_scaled_ou(noice_pred).cpu().data.numpy()
denoice_voice = m_amp_db_audio - inv_sca_noice_pred[:,0,:,:]
numpy_denoice_voice = spec_to_numpy(denoice_voice, m_pha_audio)

num_samples = numpy_denoice_voice.shape[0]
#Save all frames in one file
denoise_long = numpy_denoice_voice.reshape(1, num_samples * frame_length)*10
librosa.output.write_wav('../denoice.wav', denoise_long[0, :], sample_rate)
