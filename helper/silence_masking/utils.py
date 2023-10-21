from scipy.signal import hann
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FS = 24000
FMS = int(FS*0.1)

# All audio inputs: #channels x #frames

def load_file(_output_format_file):
    _output_dict = {}
    _fid = open(_output_format_file, 'r')
    # next(_fid)
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        if len(_words) == 5: #polar coordinates 
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
        elif len(_words) == 6: # cartesian coordinates
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])])
    _fid.close()
    return _output_dict

def get_cons_zero_list(events):
    zero_list = []
    cons_0 = []
    for frame in range(max(events.keys())+1):
        if frame not in events:
            cons_0.append(frame)
        else:
            if len(cons_0)>0:
                if len(cons_0)>1:
                    zero_list.append([cons_0[0],cons_0[-1]]) 
                else:
                    zero_list.append([cons_0[0],cons_0[0]]) 
            cons_0 = []
    return zero_list

def get_cons_ov_list(events):
    ov_list = []
    cons_ov = []
    for frame in range(max(events.keys())+1):
        if (frame in events) and len(events[frame])>1:
            cons_ov.append(frame)
        else:
            if len(cons_ov)>0:
                if len(cons_ov)>1:
                    ov_list.append([cons_ov[0],cons_ov[-1]]) 
                else:
                    ov_list.append([cons_ov[0],cons_ov[0]]) 
            cons_ov = []
    return ov_list

def int_to_frames(interval):
    f_s,f_e = interval[0], interval[1]
    f_s = f_s*FMS
    f_e = (f_e+1)*FMS
    return [f_s, f_e]

def get_audio_chunk(interval, audio):
    frame_int = int_to_frames(interval)
    f_s,f_e = frame_int[0], frame_int[1] 
    return audio[:,f_s:f_e]

def plot_audio(audio):
    # audio input : channels vs sample
    # Plot each channel separately
    fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True, sharey=True)
    for i in range(4):
        axs[i].plot(audio[i,:])
        axs[i].set_title(f'Channel {i+1}')
        axs[i].set_xlabel('Sample')
        axs[i].set_ylabel('Amplitude')
    plt.tight_layout()
    plt.show()


def smooth_audio_edge(audio):
    win_sec = 0.1
    win_samples = int(FS * win_sec)
    hann_win = hann(win_samples)

    first_half_win = hann_win[:win_samples//2]
    second_half_win = hann_win[win_samples//2:]
    audio = audio.astype(float)
    audio[:,:win_samples//2] *= first_half_win[np.newaxis,:]
    audio[:, -win_samples//2:] *= second_half_win[np.newaxis,:]
    return audio

def smooth_audio_one_edge(audio_in, use_half):
    # use_half E ['first', 'second'] half of the hann
    #assume all the audio is greater than 50 ms
    win_sec = 0.1
    win_samples = int(FS * win_sec)
    hann_win = hann(win_samples)

    first_half_win = hann_win[:win_samples//2]
    second_half_win = hann_win[win_samples//2:]
    audio = audio_in.copy()
    audio = audio.astype(float)
    if use_half=='first':
        audio *= first_half_win[np.newaxis,:]
    else:
        audio *= second_half_win[np.newaxis,:]
    
    return audio

def add_normal_noise(audio):
    mean = 0
    std = 1e-2
    noise = np.random.normal(mean, std, audio.shape)
    noisy_audio = audio + noise
    return noisy_audio

def adjust_filler(f_audio, interval):
    int_frames = int_to_frames(interval)
    fill_len = f_audio.shape[1]
    int_len = (int_frames[1]-int_frames[0])

    if fill_len > int_len :
        start = np.random.randint(0, fill_len - int_len)
        return f_audio[:,start:start+int_len] 

    elif fill_len == int_len:
        return f_audio

    else:
        num_repeats = int_len // fill_len + 1
        audio = np.tile(f_audio, (1, num_repeats))[:,:int_len]
        return audio

def mask_ov(audio_in, ov_ints, f_audio):
    audio = audio_in.copy()
    for ov_int in ov_ints:
        adj_f_aud = adjust_filler(f_audio, ov_int)
        smooth_adj_f_aud = smooth_audio_edge(adj_f_aud)
        ov_frames = int_to_frames(ov_int)
        #smooth before the mask
        if ov_frames[0] - FMS//2>=0:
            bf_mask = audio[:,(ov_frames[0] - FMS//2):ov_frames[0]]
            audio[:,(ov_frames[0] - FMS//2):ov_frames[0]] = smooth_audio_one_edge(bf_mask, 'second')
        #smooth after the mask
        if ov_frames[1] + FMS//2<audio.shape[1]:
            af_mask = audio[:,ov_frames[1]:(ov_frames[1] + FMS//2)]
            audio[:,ov_frames[1]:(ov_frames[1] + FMS//2)] = smooth_audio_one_edge(af_mask, 'first')
        
        audio[:,ov_frames[0]:ov_frames[1]] = smooth_adj_f_aud
    return audio

def remove_ov_and_save_meta(input_path, output_path):
    # return T/F if the masked file is non-empty
    df = pd.read_csv(input_path, header=None)    
    counts = df[0].value_counts()
    df_unique = df[~df[0].isin(counts[counts > 1].index)]
    df_unique[5] = df_unique[5].apply(lambda x: round(x*1e-2, 2))
    
    if len(df_unique)>0:
        df_unique.to_csv(output_path, header=False, index=False)
        return True
    else:
        return False
