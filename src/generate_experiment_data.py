import librosa 
import os 
import numpy as np
from data_tools import * 

def cal_audio_length():
    data_file = "../Train/clean_voice"
    clean_voice_list = os.listdir(data_file)
    total = 0
    for audio in clean_voice_list:
        filepath = os.path.join(data_file,audio)
        total += librosa.get_duration(filename=filepath)
    return total

def gathering_data():
    root = '../Data/clean/LibriSpeech1/test-clean'
    root_list = os.listdir(root)
    for root2 in root_list:
        tmp = os.path.join(root,root2)
        root_list2 = os.listdir(tmp)
        for root3 in root_list2:
            tmp2 = os.path.join(tmp,root3)
            root_list3 = os.listdir(tmp2)
            for x in root_list3:
                if(x[-5:] != '.flac'):
                    continue 
                tmp3 = os.path.join(tmp2,x)
                tmp4 = os.path.join('../Train/clean_voice/',x)
                os.rename(tmp3,tmp4)
    root = '../Data/clean/LibriSpeech2/dev-clean'
    root_list = os.listdir(root)
    for root2 in root_list:
        tmp = os.path.join(root,root2)
        root_list2 = os.listdir(tmp)
        for root3 in root_list2:
            tmp2 = os.path.join(tmp,root3)
            root_list3 = os.listdir(tmp2)
            for x in root_list3:
                if(x[-5:] != '.flac'):
                    continue 
                tmp3 = os.path.join(tmp2,x)
                tmp4 = os.path.join('../Train/clean_voice/',x)
                os.rename(tmp3,tmp4)            

clean_voice_dir = '../Experiment/clean_voice'
noise_voice_dir = '../Experiment/noise_voice'
sample_rate = 8000
frame_length = 8064
stride_noise = 5000
stride_clean = 8064
stride_fft = 63
min_len = 1.0
num_sample = 50
noise = audio_to_numpy(noise_voice_dir, sample_rate, frame_length, stride_noise, min_len) 
clean = audio_to_numpy(clean_voice_dir, sample_rate, frame_length, stride_clean, min_len)

## blend voice and noice together 
blend_voice = np.zeros((num_sample, frame_length))
blend_noise = np.zeros((num_sample, frame_length))
blend_noisy_voice = np.zeros((num_sample, frame_length))

for idx in range(num_sample):
    idx_voice = np.random.randint(0, clean.shape[0])
    idx_noise = np.random.randint(0, noise.shape[0])
    percent_noise = np.random.uniform(0.2, 0.8)
    blend_voice[idx, :] = clean[idx_voice, :]
    blend_noise[idx, :] = noise[idx_noise, :] * percent_noise
    blend_noisy_voice[idx, :] = blend_voice[idx, :] + blend_noise[idx, :] 


noisy_voice = blend_noisy_voice.reshape(1, num_sample * frame_length)
librosa.output.write_wav('../Experiment/sound/' + 'noisy_voice.wav', noisy_voice[0, :], sample_rate)
voice = blend_voice.reshape(1, num_sample * frame_length)
librosa.output.write_wav('../Experiment/sound/' + 'voice.wav', voice[0, :], sample_rate)
noise = blend_noise.reshape(1, num_sample * frame_length)
librosa.output.write_wav('../Experiment/sound/' + 'noise.wav', noise[0, :], sample_rate)

amp_voice, pha_voice = numpy_to_spec(blend_voice, stride_fft)
amp_noise, pha_noise = numpy_to_spec(blend_noise, stride_fft)
amp_noisy_voice, pha_noisy_voice = numpy_to_spec(blend_noisy_voice, stride_fft)

np.save('../Experiment/spectrogram/' + 'voice_amp_db', amp_voice)
np.save('../Experiment/spectrogram/' + 'noise_amp_db', amp_noise)
np.save('../Experiment/spectrogram/' + 'noisy_voice_amp_db', amp_noisy_voice)

np.save('../Experiment/spectrogram/' + 'voice_pha_db', pha_voice)
np.save('../Experiment/spectrogram/' + 'noise_pha_db', pha_noise)
np.save('../Experiment/spectrogram/' + 'noisy_voice_pha_db', pha_noisy_voice)
