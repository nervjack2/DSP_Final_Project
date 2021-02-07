import librosa 
import os 
import numpy as np

def stft_to_audio(mag, pha):
    #print('Short time furier transform back to audio in time series')
    frame_length = 8064
    hop_length_fft = 63
    mag_amp = librosa.db_to_amplitude(mag, ref=1.0)
    audio_stft = mag_amp * pha
    audio_reconstruct = librosa.core.istft(audio_stft, hop_length=hop_length_fft, length=frame_length)
    return audio_reconstruct

def spec_to_numpy(mag, phase):
    print('Transform spectrogram to numpy array')
    list_audio = []
    nb_spec = mag.shape[0]
    for i in range(nb_spec):
        audio_reconstruct = stft_to_audio(mag[i], phase[i])
        list_audio.append(audio_reconstruct)
    return np.vstack(list_audio)

def audio_to_numpy(voice_dir, sample_rate, frame_length, hop_length, min_len):
    print('Transform audio to numpy array')
    sound_list = []
    list_voice_dir = os.listdir(voice_dir)
    for i,audio in enumerate(list_voice_dir):
        print("{}/{}".format(i,len(list_voice_dir)))
        y, sr = librosa.load(os.path.join(voice_dir, audio), sr=sample_rate)
        audio_len = librosa.get_duration(y=y, sr=sr)
        if(min_len <= audio_len):
            audio_sample_len = y.shape[0]
            split_audio_list = []
            for start in range(0, audio_sample_len-frame_length+1, hop_length):
                split_audio_list.append(y[start:start+frame_length])
            sound_list.append(np.vstack(split_audio_list))
    return np.vstack(sound_list)

def numpy_to_spec(numpy_audio, hop_length_fft):
    print('Transform numpy array to spectrogram')
    num_audio = numpy_audio.shape[0]
    nfft = 255
    spec_dim = int(255/2)+1
    amp_db = np.zeros((num_audio, spec_dim, spec_dim))
    ph = np.zeros((num_audio, spec_dim, spec_dim), dtype=complex)
    for i in range(num_audio):
        print("{}/{}".format(i,num_audio))
        stft_audio = librosa.stft(numpy_audio[i], n_fft=nfft, hop_length=hop_length_fft)
        stft_amp, stft_ph = librosa.magphase(stft_audio)
        stft_amp_db = librosa.amplitude_to_db(stft_amp, ref=np.max)
        amp_db[i, :, :] = stft_amp_db
        ph[i, :, :] = stft_ph
    return amp_db, ph

def scaled_in(spec):
    return (spec + 46)/50

def scaled_ou(spec):
    return (spec - 6)/82

def inv_scaled_in(spec):
    return spec * 50 - 46

def inv_scaled_ou(spec):
    return spec * 82 + 6
