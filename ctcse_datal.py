import glob
import os
import librosa
import numpy as np
import time
from utils import for_stft_2

from hparam import hparam as hp

print('start to prepare data for ctc')

rms = lambda y: np.sqrt(np.mean(np.square(y), axis=-1))
zero_audio = np.zeros(3*hp.data.sr)
error_num = 0
def create_mapping(map_file):
   state2int = {}
   count = 1
   with open(map_file) as m:     # setup the mapping from state to integer
      for line in m:
         line = line.strip()
         if line:
            state2int[line] = count
            count += 1
   return state2int

def change_audio(utterance_path):
    target_name = utterance_path.split(".")[0]
    state2int = create_mapping("./hmmlist")
    target_file = "%s.PHN" % target_name
    audio_clean, _ = librosa.load(utterance_path, sr=hp.data.sr)
    # print('audio_clean:', audio_clean)
    len_speech = len(audio_clean)
    zero_audio = np.zeros(3*hp.data.sr)
    target_return = []
    if len_speech <= (3 * hp.data.sr):
        # n_repeat = int(np.ceil(float(3 * hp.data.sr) / float(len_speech)))
        # audio_clean_mi = np.tile(audio_clean, n_repeat)
        # audio_clean = audio_clean_mi[0: 3 * hp.data.sr]
        zero_audio[0: len_speech] = audio_clean
        with open(target_file, 'r') as f:
            for line in f:
                line = line.strip()
                _, _, state = line.split()[:3]
                label = state2int[state]
                target_return.append(label)
    else:
        # clean_onset = rs.randint(0, len_speech - 3 * hp.data.sr, 1)[0]
        # clean_offset = clean_onset + 3 * hp.data.sr
        # zero_audio = audio_clean[0: 3*hp.data.sr]
        with open(target_file, 'r') as f:
            for line in f:
                line = line.strip()
                start, end, state = line.split()[:3]
                # print('start:', start)
                # print('end:', end)
                # print('state:', state)
                end = int(end)
                start = int(start)
                if end < 3*hp.data.sr:
                    label = state2int[state]
                    target_return.append(label)
                else:
                    zero_audio[0: start] = audio_clean[0:start]
                    # print(start)
                    break
    return zero_audio, target_return

audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))
total_speaker_num  = len(audio_path)
train_speaker_num = (total_speaker_num//10)*9

rs = np.random.RandomState(0)

for i, speaker in enumerate(audio_path):
    for j, audio_name in enumerate(os.listdir(speaker)):
        if audio_name[-4:] == '.WAV':
            # print('audio_name:', audio_name)
            utterance_path = os.path.join(speaker, audio_name)
            # print('utterance_path:', utterance_path)
            # audio_clean, sr = librosa.load(utterance_path, sr=hp.data.sr)
            audio_clean, target_clean = change_audio(utterance_path)
            if (all(audio_clean == zero_audio)):
                error_num = error_num+1
                print('error_num:', error_num)
                break
            # print('audio_clean:', audio_clean)
            # print('target_clean:', target_clean)
            len_speech = 3*hp.data.sr
            # packet_audio = audio_clean
            if hp.data.need_noise:
                noise_names = []
                for noise_name in os.listdir(hp.data.noise_path_all):
                    noise_names.append(noise_name)
                rand_noise_names_15 = rs.choice(noise_names, hp.data.num_noise_utterance_15)
                rand_noise_names_10 = rs.choice(noise_names, hp.data.num_noise_utterance_10)
                rand_noise_names_5 = rs.choice(noise_names, hp.data.num_noise_utterance_5)
                rand_noise_names_0 = rs.choice(noise_names, hp.data.num_noise_utterance_0)
                rand_noise_names__5 = rs.choice(noise_names, hp.data.num_noise_utterance__5)
                for k, rand_noise_name in enumerate(rand_noise_names_15):
                    # print('noise_name:', rand_noise_name)
                    # print('noise_name:', rand_noise_name)
                    rand_noise_audio_path = os.path.join(hp.data.noise_path_all, rand_noise_name)
                    noise_audio, sr = librosa.load(rand_noise_audio_path, sr=hp.data.sr)
                    len_noise = len(noise_audio)
                    # print('len_noise:', len_noise)
                    # print('len_speech:', len(audio_clean))
                    if len_noise <= 3*hp.data.sr:
                        n_repeat = int(np.ceil(float(len_speech) / float(len_noise)))
                        noise_audio_mi = np.tile(noise_audio, n_repeat)
                        noise_audio_mix = noise_audio_mi[0: len_speech]
                        noise_onset = 0
                        noise_offset = len_speech
                    else:
                        noise_onset = rs.randint(0, len_noise - len_speech, 1)[0]
                        noise_offset = noise_onset + len_speech
                        noise_audio_mix = noise_audio[noise_onset: noise_offset]
                    # print('rms_clean:', rms(audio_clean))
                    # print('rms_noise:', rms(noise_audio_mix))
                    # print('shape:', np.shape(rms(audio_clean)))
                    # print('shape_audio:', np.shape(rms(audio_clean)))
                    # print('audio_clean_change:', audio_clean)
                    # print('noise_audio_mix:', noise_audio_mix)
                    original_snr = rms(audio_clean) / rms(noise_audio_mix)
                    # original_snr = np.mean(original_snr)
                    # print('original_snr:', original_snr)
                    target_snr = 10. ** (float(15) / 20.)
                    # print('target_snr:', target_snr)
                    clean_scaling = target_snr / original_snr
                    # print('clean_scaling:', clean_scaling)
                    audio_clean_mix = audio_clean
                    audio_clean_mix = audio_clean_mix * clean_scaling
                    audio_mix = audio_clean_mix + noise_audio_mix
                    alpha = 0.9 / np.max(np.abs(audio_mix))
                    audio_mix = audio_mix * alpha
                    audio_clean_mix = audio_clean_mix * alpha
                    audio_mix_tf = np.transpose(for_stft_2(audio_mix), (1,0))
                    # print('audio_mix_tf.shape:', audio_mix_tf.shape)
                    audio_clean_tf = np.transpose(for_stft_2(audio_clean_mix), (1,0))
                    # print('audio_clean_tf.shape:', audio_clean_tf.shape)
                    if i < train_speaker_num:
                        # file_path_clean = os.path.join("/data2/lps/data_ctc/train", "speaker%s_utterance%d_noise%d_5_clean.wav"%(i, j, k))
                        file_path = os.path.join("/data2/lps/data_ctc/train", "speaker%s_utterance%d_noise%d_15.npy"%(i, j, k))
                        # file_path_noisy = os.path.join("/data2/lps/data_ctc/train", "speaker%s_utterance%d_noise%d_15_noisy.wav"%(i, j, k))
                        out_file = []
                        out_file.append(audio_mix_tf)
                        out_file.append(target_clean)
                        out_file.append(audio_clean_tf)
                        # print(out_file[2])
                        # librosa.output.write_wav(file_path_noisy, audio_mix, sr=16000)
                        # librosa.output.write_wav(file_path_clean, audio_clean_mix, sr=16000)
                        np.save(file_path, out_file)
                    else:
                        file_path = os.path.join("/data2/lps/data_ctc/test", "speaker%s_utterance%d_noise%d_15.npy"%(i, j, k))
                        # file_path_noisy = os.path.join("/data2/lps/data_ctc/train", "speaker%s_utterance%d_noise%d_15_noisy.wav"%(i, j, k))
                        out_file = []
                        out_file.append(audio_mix_tf)
                        out_file.append(target_clean)
                        out_file.append(audio_clean_tf)
                        out_file.append(audio_mix)
                        out_file.append(audio_clean_mix)
                        # print(out_file[2])
                        # librosa.output.write_wav(file_path_noisy, audio_mix, sr=16000)
                        # librosa.output.write_wav(file_path_clean, audio_clean_mix, sr=16000)
                        np.save(file_path, out_file)
                for k, rand_noise_name in enumerate(rand_noise_names_10):
                    # print('noise_name:', rand_noise_name)
                    # print('noise_name:', rand_noise_name)
                    rand_noise_audio_path = os.path.join(hp.data.noise_path_all, rand_noise_name)
                    noise_audio, sr = librosa.load(rand_noise_audio_path, sr=hp.data.sr)
                    len_noise = len(noise_audio)
                    # print('len_noise:', len_noise)
                    # print('len_speech:', len(audio_clean))
                    if len_noise <= 3*hp.data.sr:
                        n_repeat = int(np.ceil(float(len_speech) / float(len_noise)))
                        noise_audio_mi = np.tile(noise_audio, n_repeat)
                        noise_audio_mix = noise_audio_mi[0: len_speech]
                        noise_onset = 0
                        noise_offset = len_speech
                    else:
                        noise_onset = rs.randint(0, len_noise - len_speech, 1)[0]
                        noise_offset = noise_onset + len_speech
                        noise_audio_mix = noise_audio[noise_onset: noise_offset]
                    # print('rms_clean:', rms(audio_clean))
                    # print('rms_noise:', rms(noise_audio_mix))
                    # print('shape:', np.shape(rms(audio_clean)))
                    # print('shape_audio:', np.shape(rms(audio_clean)))
                    original_snr = rms(audio_clean) / rms(noise_audio_mix)
                    # original_snr = np.mean(original_snr)
                    # print('original_snr:', original_snr)
                    target_snr = 10. ** (float(10) / 20.)
                    # print('target_snr:', target_snr)
                    clean_scaling = target_snr / original_snr
                    # print('clean_scaling:', clean_scaling)
                    audio_clean_mix = audio_clean
                    audio_clean_mix = audio_clean_mix * clean_scaling
                    audio_mix = audio_clean_mix + noise_audio_mix
                    alpha = 0.9 / np.max(np.abs(audio_mix))
                    audio_mix = audio_mix * alpha
                    audio_clean_mix = audio_clean_mix * alpha
                    audio_mix_tf = np.transpose(for_stft_2(audio_mix), (1, 0))
                    # print('audio_mix_tf.shape:', audio_mix_tf.shape)
                    audio_clean_tf = np.transpose(for_stft_2(audio_clean_mix), (1, 0))
                    # print('audio_clean_tf.shape:', audio_clean_tf.shape)
                    if i < train_speaker_num:
                        # file_path_clean = os.path.join("/data2/lps/data_ctc/train", "speaker%s_utterance%d_noise%d_5_clean.wav"%(i, j, k))
                        file_path = os.path.join("/data2/lps/data_ctc/train",
                                                 "speaker%s_utterance%d_noise%d_10.npy" % (i, j, k))
                        # file_path_noisy = os.path.join("/data2/lps/data_ctc/train", "speaker%s_utterance%d_noise%d_15_noisy.wav"%(i, j, k))
                        out_file = []
                        out_file.append(audio_mix_tf)
                        out_file.append(target_clean)
                        out_file.append(audio_clean_tf)
                        # print(out_file[2])
                        # librosa.output.write_wav(file_path_noisy, audio_mix, sr=16000)
                        # librosa.output.write_wav(file_path_clean, audio_clean_mix, sr=16000)
                        np.save(file_path, out_file)
                    else:
                        file_path = os.path.join("/data2/lps/data_ctc/test",
                                                 "speaker%s_utterance%d_noise%d_10.npy" % (i, j, k))
                        # file_path_noisy = os.path.join("/data2/lps/data_ctc/train", "speaker%s_utterance%d_noise%d_15_noisy.wav"%(i, j, k))
                        out_file = []
                        out_file.append(audio_mix_tf)
                        out_file.append(target_clean)
                        out_file.append(audio_clean_tf)
                        out_file.append(audio_mix)
                        out_file.append(audio_clean_mix)
                        # print(out_file[2])
                        # librosa.output.write_wav(file_path_noisy, audio_mix, sr=16000)
                        # librosa.output.write_wav(file_path_clean, audio_clean_mix, sr=16000)
                        np.save(file_path, out_file)
                for k, rand_noise_name in enumerate(rand_noise_names_5):
                    # print('noise_name:', rand_noise_name)
                    # print('noise_name:', rand_noise_name)
                    rand_noise_audio_path = os.path.join(hp.data.noise_path_all, rand_noise_name)
                    noise_audio, sr = librosa.load(rand_noise_audio_path, sr=hp.data.sr)
                    len_noise = len(noise_audio)
                    # print('len_noise:', len_noise)
                    # print('len_speech:', len(audio_clean))
                    if len_noise <= 3*hp.data.sr:
                        n_repeat = int(np.ceil(float(len_speech) / float(len_noise)))
                        noise_audio_mi = np.tile(noise_audio, n_repeat)
                        noise_audio_mix = noise_audio_mi[0: len_speech]
                        noise_onset = 0
                        noise_offset = len_speech
                    else:
                        noise_onset = rs.randint(0, len_noise - len_speech, 1)[0]
                        noise_offset = noise_onset + len_speech
                        noise_audio_mix = noise_audio[noise_onset: noise_offset]
                    # print('rms_clean:', rms(audio_clean))
                    # print('rms_noise:', rms(noise_audio_mix))
                    # print('shape:', np.shape(rms(audio_clean)))
                    # print('shape_audio:', np.shape(rms(audio_clean)))
                    original_snr = rms(audio_clean) / rms(noise_audio_mix)
                    # original_snr = np.mean(original_snr)
                    # print('original_snr:', original_snr)
                    target_snr = 10. ** (float(5) / 20.)
                    # print('target_snr:', target_snr)
                    clean_scaling = target_snr / original_snr
                    # print('clean_scaling:', clean_scaling)
                    audio_clean_mix = audio_clean
                    audio_clean_mix = audio_clean_mix * clean_scaling
                    audio_mix = audio_clean_mix + noise_audio_mix
                    alpha = 0.9 / np.max(np.abs(audio_mix))
                    audio_mix = audio_mix * alpha
                    audio_clean_mix = audio_clean_mix * alpha
                    audio_mix_tf = np.transpose(for_stft_2(audio_mix), (1, 0))
                    # print('audio_mix_tf.shape:', audio_mix_tf.shape)
                    audio_clean_tf = np.transpose(for_stft_2(audio_clean_mix), (1, 0))
                    # print('audio_clean_tf.shape:', audio_clean_tf.shape)
                    if i < train_speaker_num:
                        # file_path_clean = os.path.join("/data2/lps/data_ctc/train", "speaker%s_utterance%d_noise%d_5_clean.wav"%(i, j, k))
                        file_path = os.path.join("/data2/lps/data_ctc/train",
                                                 "speaker%s_utterance%d_noise%d_5.npy" % (i, j, k))
                        # file_path_noisy = os.path.join("/data2/lps/data_ctc/train", "speaker%s_utterance%d_noise%d_15_noisy.wav"%(i, j, k))
                        out_file = []
                        out_file.append(audio_mix_tf)
                        out_file.append(target_clean)
                        out_file.append(audio_clean_tf)
                        # print(out_file[2])
                        # librosa.output.write_wav(file_path_noisy, audio_mix, sr=16000)
                        # librosa.output.write_wav(file_path_clean, audio_clean_mix, sr=16000)
                        np.save(file_path, out_file)
                    else:
                        file_path = os.path.join("/data2/lps/data_ctc/test",
                                                 "speaker%s_utterance%d_noise%d_5.npy" % (i, j, k))
                        # file_path_noisy = os.path.join("/data2/lps/data_ctc/train", "speaker%s_utterance%d_noise%d_15_noisy.wav"%(i, j, k))
                        out_file = []
                        out_file.append(audio_mix_tf)
                        out_file.append(target_clean)
                        out_file.append(audio_clean_tf)
                        out_file.append(audio_mix)
                        out_file.append(audio_clean_mix)
                        # print(out_file[2])
                        # librosa.output.write_wav(file_path_noisy, audio_mix, sr=16000)
                        # librosa.output.write_wav(file_path_clean, audio_clean_mix, sr=16000)
                        np.save(file_path, out_file)
                for k, rand_noise_name in enumerate(rand_noise_names_0):
                    # print('noise_name:', rand_noise_name)
                    # print('noise_name:', rand_noise_name)
                    rand_noise_audio_path = os.path.join(hp.data.noise_path_all, rand_noise_name)
                    noise_audio, sr = librosa.load(rand_noise_audio_path, sr=hp.data.sr)
                    len_noise = len(noise_audio)
                    # print('len_noise:', len_noise)
                    # print('len_speech:', len(audio_clean))
                    if len_noise <= 3*hp.data.sr:
                        n_repeat = int(np.ceil(float(len_speech) / float(len_noise)))
                        noise_audio_mi = np.tile(noise_audio, n_repeat)
                        noise_audio_mix = noise_audio_mi[0: len_speech]
                        noise_onset = 0
                        noise_offset = len_speech
                    else:
                        noise_onset = rs.randint(0, len_noise - len_speech, 1)[0]
                        noise_offset = noise_onset + len_speech
                        noise_audio_mix = noise_audio[noise_onset: noise_offset]
                    # print('rms_clean:', rms(audio_clean))
                    # print('rms_noise:', rms(noise_audio_mix))
                    # print('shape:', np.shape(rms(audio_clean)))
                    # print('shape_audio:', np.shape(rms(audio_clean)))
                    original_snr = rms(audio_clean) / rms(noise_audio_mix)
                    # original_snr = np.mean(original_snr)
                    # print('original_snr:', original_snr)
                    target_snr = 10. ** (float(0) / 20.)
                    # print('target_snr:', target_snr)
                    clean_scaling = target_snr / original_snr
                    # print('clean_scaling:', clean_scaling)
                    audio_clean_mix = audio_clean
                    audio_clean_mix = audio_clean_mix * clean_scaling
                    audio_mix = audio_clean_mix + noise_audio_mix
                    alpha = 0.9 / np.max(np.abs(audio_mix))
                    audio_mix = audio_mix * alpha
                    audio_clean_mix = audio_clean_mix * alpha
                    audio_mix_tf = np.transpose(for_stft_2(audio_mix), (1, 0))
                    # print('audio_mix_tf.shape:', audio_mix_tf.shape)
                    audio_clean_tf = np.transpose(for_stft_2(audio_clean_mix), (1, 0))
                    # print('audio_clean_tf.shape:', audio_clean_tf.shape)
                    if i < train_speaker_num:
                        # file_path_clean = os.path.join("/data2/lps/data_ctc/train", "speaker%s_utterance%d_noise%d_5_clean.wav"%(i, j, k))
                        file_path = os.path.join("/data2/lps/data_ctc/train",
                                                 "speaker%s_utterance%d_noise%d_0.npy" % (i, j, k))
                        # file_path_noisy = os.path.join("/data2/lps/data_ctc/train", "speaker%s_utterance%d_noise%d_15_noisy.wav"%(i, j, k))
                        out_file = []
                        out_file.append(audio_mix_tf)
                        out_file.append(target_clean)
                        out_file.append(audio_clean_tf)
                        # print(out_file[2])
                        # librosa.output.write_wav(file_path_noisy, audio_mix, sr=16000)
                        # librosa.output.write_wav(file_path_clean, audio_clean_mix, sr=16000)
                        np.save(file_path, out_file)
                    else:
                        file_path = os.path.join("/data2/lps/data_ctc/test",
                                                 "speaker%s_utterance%d_noise%d_0.npy" % (i, j, k))
                        # file_path_noisy = os.path.join("/data2/lps/data_ctc/train", "speaker%s_utterance%d_noise%d_15_noisy.wav"%(i, j, k))
                        out_file = []
                        out_file.append(audio_mix_tf)
                        out_file.append(target_clean)
                        out_file.append(audio_clean_tf)
                        out_file.append(audio_mix)
                        out_file.append(audio_clean_mix)
                        # print(out_file[2])
                        # librosa.output.write_wav(file_path_noisy, audio_mix, sr=16000)
                        # librosa.output.write_wav(file_path_clean, audio_clean_mix, sr=16000)
                        np.save(file_path, out_file)
                for k, rand_noise_name in enumerate(rand_noise_names__5):
                    # print('noise_name:', rand_noise_name)
                    # print('noise_name:', rand_noise_name)
                    rand_noise_audio_path = os.path.join(hp.data.noise_path_all, rand_noise_name)
                    noise_audio, sr = librosa.load(rand_noise_audio_path, sr=hp.data.sr)
                    len_noise = len(noise_audio)
                    # print('len_noise:', len_noise)
                    # print('len_speech:', len(audio_clean))
                    if len_noise <= 3*hp.data.sr:
                        n_repeat = int(np.ceil(float(len_speech) / float(len_noise)))
                        noise_audio_mi = np.tile(noise_audio, n_repeat)
                        noise_audio_mix = noise_audio_mi[0: len_speech]
                        noise_onset = 0
                        noise_offset = len_speech
                    else:
                        noise_onset = rs.randint(0, len_noise - len_speech, 1)[0]
                        noise_offset = noise_onset + len_speech
                        noise_audio_mix = noise_audio[noise_onset: noise_offset]
                    # print('rms_clean:', rms(audio_clean))
                    # print('rms_noise:', rms(noise_audio_mix))
                    # print('shape:', np.shape(rms(audio_clean)))
                    # print('shape_audio:', np.shape(rms(audio_clean)))
                    original_snr = rms(audio_clean) / rms(noise_audio_mix)
                    # original_snr = np.mean(original_snr)
                    # print('original_snr:', original_snr)
                    target_snr = 10. ** (float(-5) / 20.)
                    # print('target_snr:', target_snr)
                    clean_scaling = target_snr / original_snr
                    # print('clean_scaling:', clean_scaling)
                    audio_clean_mix = audio_clean
                    audio_clean_mix = audio_clean_mix * clean_scaling
                    audio_mix = audio_clean_mix + noise_audio_mix
                    alpha = 0.9 / np.max(np.abs(audio_mix))
                    audio_mix = audio_mix * alpha
                    audio_clean_mix = audio_clean_mix * alpha
                    audio_mix_tf = np.transpose(for_stft_2(audio_mix), (1, 0))
                    # print('audio_mix_tf.shape:', audio_mix_tf.shape)
                    audio_clean_tf = np.transpose(for_stft_2(audio_clean_mix), (1, 0))
                    # print('audio_clean_tf.shape:', audio_clean_tf.shape)
                    if i < train_speaker_num:
                        # file_path_clean = os.path.join("/data2/lps/data_ctc/train", "speaker%s_utterance%d_noise%d_5_clean.wav"%(i, j, k))
                        file_path = os.path.join("/data2/lps/data_ctc/train",
                                                 "speaker%s_utterance%d_noise%d__5.npy" % (i, j, k))
                        # file_path_noisy = os.path.join("/data2/lps/data_ctc/train", "speaker%s_utterance%d_noise%d_15_noisy.wav"%(i, j, k))
                        out_file = []
                        out_file.append(audio_mix_tf)
                        out_file.append(target_clean)
                        out_file.append(audio_clean_tf)
                        # print(out_file[2])
                        # librosa.output.write_wav(file_path_noisy, audio_mix, sr=16000)
                        # librosa.output.write_wav(file_path_clean, audio_clean_mix, sr=16000)
                        np.save(file_path, out_file)
                    else:
                        file_path = os.path.join("/data2/lps/data_ctc/test",
                                                 "speaker%s_utterance%d_noise%d__5.npy" % (i, j, k))
                        # file_path_noisy = os.path.join("/data2/lps/data_ctc/train", "speaker%s_utterance%d_noise%d_15_noisy.wav"%(i, j, k))
                        out_file = []
                        out_file.append(audio_mix_tf)
                        out_file.append(target_clean)
                        out_file.append(audio_clean_tf)
                        out_file.append(audio_mix)
                        out_file.append(audio_clean_mix)
                        # print(out_file[2])
                        # librosa.output.write_wav(file_path_noisy, audio_mix, sr=16000)
                        # librosa.output.write_wav(file_path_clean, audio_clean_mix, sr=16000)
                        np.save(file_path, out_file)