import torch
import numpy as np
import os
import glob
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
# import shutil
# import collections
# import set_model_ctc
# from warpctc_pytorch import CTCLoss
from random import sample, shuffle
import torch.nn.functional as F
import librosa
# from ctc_decode import Decoder
from hparam import hparam as hp
# import set_model_ctc
import time
from utils import gri_lim_1
from pypesq import pesq
from pystoi.stoi import stoi

layers = 4          # number of LSTM layers
hidden_size = 512   # number of cells per direction
num_dirs = 2        # number of LSTM directions

n = 0
n_20 = 0
n_10 = 0
n_15 = 0
n_5 = 0
n_0 = 0
n__5 = 0
score_20_pesq = 0
score_10_pesq = 0
score_15_pesq = 0
score_0_pesq = 0
score_5_pesq = 0
score__5_pesq = 0
score_20_stoi = 0
score_10_stoi = 0
score_15_stoi = 0
score_0_stoi = 0
score_5_stoi = 0
score__5_stoi = 0
score_20_pesq_gl = 0
score_10_pesq_gl = 0
score_15_pesq_gl = 0
score_0_pesq_gl = 0
score_5_pesq_gl = 0
score__5_pesq_gl = 0
score_20_stoi_gl = 0
score_10_stoi_gl = 0
score_15_stoi_gl = 0
score_0_stoi_gl = 0
score_5_stoi_gl = 0
score__5_stoi_gl = 0
window_length = int(hp.data.window * hp.data.sr)
hop_length = int(hp.data.hop * hp.data.sr)
# clean_audio_path = '/data2/lps/data_gan_test/clean_wav/'
# mix_audio_path = '/data2/lps/data_gan_test/noisy_wav/'
# mix_path = '/data2/lps/data_gan_test/noisy_tf/'
# clean_audio_path = '/workspace/data/rgan/test/clean_wav'
mix_audio_path = '/raid/data/lps/data_ctc/test/'

class VoiceFilter(nn.Module):
    def __init__(self):
        super(VoiceFilter, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 7), padding=(0, 3), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), padding=(3, 0), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(2, 2), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(6, 2), dilation=(3, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(10, 2), dilation=(5, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(26, 2), dilation=(13, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))

        self.LSTM1 = nn.LSTM(257 * 8, 400, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=400, hidden_size=hidden_size, bidirectional=True, bias=False,
                             batch_first=True)
        self.fc = nn.Sequential(
            # nn.BatchNorm1d(2 * rnn_hidden_size),
            nn.Linear(2 * hidden_size, 62, bias=False)
        )
        self.lstm3 = nn.LSTM(input_size=2 * hidden_size, hidden_size=hidden_size, bidirectional=True,
                             bias=False, batch_first=True)

        for name, param in self.LSTM1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.FC3 = nn.Sequential(nn.Dropout(0.1, False), nn.Linear(400, 600), nn.ReLU())
        for name, param in self.FC3.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.FC4 = nn.Sequential(nn.Dropout(0.1, False), nn.Linear(600, 257))
        for name, param in self.FC4.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.lstm2.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        for name, param in self.fc.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, noisy_tf_gpu):
        x = self.CNN(noisy_tf_gpu)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        out_real = x
        out_3, _ = self.LSTM1(out_real)
        out_3 = F.relu(out_3)
        out = self.FC3(out_3)
        out = self.FC4(out)
        mask = torch.sigmoid(out)
        out_4, _ = self.lstm2(out_3)
        out_4 = F.relu(out_4)
        out_5, _ = self.lstm3(out_4)
        out_5 = F.relu(out_5)
        out_6 = self.fc(out_5)

        return (mask, out_6)

if __name__ == '__main__':
    np_file_list = os.listdir(mix_audio_path)
    save_model = torch.load("/raid/data/lps/model/ctc/epoch_25.pth")
    voicefilter_net = VoiceFilter()
    voicefilter_net.eval()
    model_dict_vf = voicefilter_net.state_dict()

    new_state_dict = {k[7:]: v for k, v in save_model.items()}
    state_dict_vf = {k: v for k, v in new_state_dict.items() if k in model_dict_vf}
    model_dict_vf.update(state_dict_vf)
    voicefilter_net.load_state_dict(model_dict_vf)

    for item in np_file_list:
        print('item:', item)
        item_name = item.split(".")[0]
        som_th = item_name[-2: ]
        if (som_th == '15'):

            mix_file_name = "%s.npy"%item_name
            mix_file = np.load(os.path.join(mix_audio_path, mix_file_name))
            n_15 = n_15 + 1
            mix_audio_tf = mix_file[0]
            mix_audio = mix_file[3]
            clean_audio = mix_file[4]
            mix_audio_tf = torch.Tensor(mix_audio_tf)
            mix_tf_0 = librosa.stft(mix_audio, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            mix_audio_tf_1 = mix_audio_tf.unsqueeze(0)
            mix_audio_tf_1 = mix_audio_tf_1.unsqueeze(0)

            mask, _ = voicefilter_net(mix_audio_tf_1)
            de_tf = mix_audio_tf * mask
            aaaa = torch.squeeze(de_tf)
            aaaa = aaaa.transpose(0, 1).detach().numpy()
            angle_ = np.angle(mix_tf_0)
            aaa = librosa.istft(aaaa*(np.exp(1j*angle_)), hop_length=hop_length, win_length=window_length)

            out = gri_lim_1(aaaa, mix_audio, hp.data.nfft, hop_length, window_length)
            score_pesq_gl = pesq(clean_audio, out, 16000)
            score_stoi_gl = stoi(clean_audio, out, 16000)
            score_pesq = pesq(clean_audio, aaa, 16000)
            score_stoi = stoi(clean_audio, aaa, 16000)
            score_15_pesq = score_15_pesq + score_pesq
            score_15_stoi = score_15_stoi + score_stoi
            score_15_pesq_gl = score_15_pesq_gl + score_pesq_gl
            score_15_stoi_gl = score_15_stoi_gl + score_stoi_gl

            print('n_15', n_15, 'score_15_pesq:', score_15_pesq, 'val_pesq:', score_15_pesq / n_15)
            print('n_15', n_15, 'score_15_stoi:', score_15_stoi, 'val_stoi:', score_15_stoi / n_15)
            print('n_15', n_15, 'score_15_pesq_gl:', score_15_pesq_gl, 'val_pesq_gl:', score_15_pesq_gl / n_15)
            print('n_15', n_15, 'score_15_stoi_gl:', score_15_stoi_gl, 'val_stoi_gl:', score_15_stoi_gl / n_15)
        elif(som_th == '10'):
            mix_file_name = "%s.npy" % item_name
            mix_file = np.load(os.path.join(mix_audio_path, mix_file_name))
            n_10 = n_10 + 1
            mix_audio_tf = mix_file[0]
            mix_audio = mix_file[3]
            clean_audio = mix_file[4]
            mix_audio_tf = torch.Tensor(mix_audio_tf)
            mix_tf_0 = librosa.stft(mix_audio, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            mix_audio_tf_1 = mix_audio_tf.unsqueeze(0)
            mix_audio_tf_1 = mix_audio_tf_1.unsqueeze(0)
            mask, _ = voicefilter_net(mix_audio_tf_1)
            de_tf = mix_audio_tf * mask
            aaaa = torch.squeeze(de_tf)
            aaaa = aaaa.transpose(0, 1).detach().numpy()
            angle_ = np.angle(mix_tf_0)
            aaa = librosa.istft(aaaa*(np.exp(1j*angle_)), hop_length=hop_length, win_length=window_length)
            out = gri_lim_1(aaaa, mix_audio, hp.data.nfft, hop_length, window_length)
            score_pesq_gl = pesq(clean_audio, out, 16000)
            score_stoi_gl = stoi(clean_audio, out, 16000)
            score_pesq = pesq(clean_audio, aaa, 16000)
            score_stoi = stoi(clean_audio, aaa, 16000)
            score_10_pesq = score_10_pesq + score_pesq
            score_10_stoi = score_10_stoi + score_stoi
            score_10_pesq_gl = score_10_pesq_gl + score_pesq_gl
            score_10_stoi_gl = score_10_stoi_gl + score_stoi_gl

            print('n_10', n_10, 'score_10_pesq:', score_10_pesq, 'val_pesq:', score_10_pesq / n_10)
            print('n_10', n_10, 'score_10_stoi:', score_10_stoi, 'val_stoi:', score_10_stoi / n_10)
            print('n_10', n_10, 'score_10_pesq_gl:', score_10_pesq_gl, 'val_pesq_gl:', score_10_pesq_gl / n_10)
            print('n_10', n_10, 'score_10_stoi_gl:', score_10_stoi_gl, 'val_stoi_gl:', score_10_stoi_gl / n_10)
        elif(som_th == '_0'):
            mix_file_name = "%s.npy" % item_name
            mix_file = np.load(os.path.join(mix_audio_path, mix_file_name))
            n_0 = n_0 + 1
            mix_audio_tf = mix_file[0]
            mix_audio = mix_file[3]
            clean_audio = mix_file[4]
            mix_audio_tf = torch.Tensor(mix_audio_tf)
            mix_tf_0 = librosa.stft(mix_audio, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
            mix_audio_tf_1 = mix_audio_tf.unsqueeze(0)
            mix_audio_tf_1 = mix_audio_tf_1.unsqueeze(0)
            mask, _ = voicefilter_net(mix_audio_tf_1)
            de_tf = mix_audio_tf * mask
            aaaa = torch.squeeze(de_tf)
            aaaa = aaaa.transpose(0, 1).detach().numpy()
            angle_ = np.angle(mix_tf_0)
            aaa = librosa.istft(aaaa*(np.exp(1j*angle_)), hop_length=hop_length, win_length=window_length)
            out = gri_lim_1(aaaa, mix_audio, hp.data.nfft, hop_length, window_length)
            score_pesq_gl = pesq(clean_audio, out, 16000)
            score_stoi_gl = stoi(clean_audio, out, 16000)
            score_pesq = pesq(clean_audio, aaa, 16000)
            score_stoi = stoi(clean_audio, aaa, 16000)
            score_0_pesq = score_0_pesq + score_pesq
            score_0_stoi = score_0_stoi + score_stoi
            score_0_pesq_gl = score_0_pesq_gl + score_pesq_gl
            score_0_stoi_gl = score_0_stoi_gl + score_stoi_gl

            print('n_0', n_0, 'score_0_pesq:', score_0_pesq, 'val_pesq:', score_0_pesq / n_0)
            print('n_0', n_0, 'score_0_stoi:', score_0_stoi, 'val_stoi:', score_0_stoi / n_0)
            print('n_0', n_0, 'score_0_pesq_gl:', score_0_pesq_gl, 'val_pesq_gl:', score_0_pesq_gl / n_0)
            print('n_0', n_0, 'score_0_stoi_gl:', score_0_stoi_gl, 'val_stoi_gl:', score_0_stoi_gl / n_0)
        elif (som_th == '_5'):
            som_th_1 = item_name[-3:]
            if (som_th_1 == '__5'):
                mix_file_name = "%s.npy" % item_name
                mix_file = np.load(os.path.join(mix_audio_path, mix_file_name))
                n__5 = n__5 + 1
                mix_audio_tf = mix_file[0]
                mix_audio = mix_file[3]
                clean_audio = mix_file[4]
                mix_audio_tf = torch.Tensor(mix_audio_tf)
                mix_tf_0 = librosa.stft(mix_audio, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
                mix_audio_tf_1 = mix_audio_tf.unsqueeze(0)
                mix_audio_tf_1 = mix_audio_tf_1.unsqueeze(0)
                mask, _ = voicefilter_net(mix_audio_tf_1)
                de_tf = mix_audio_tf * mask
                aaaa = torch.squeeze(de_tf)
                aaaa = aaaa.transpose(0, 1).detach().numpy()
                angle_ = np.angle(mix_tf_0)
                aaa = librosa.istft(aaaa * (np.exp(1j * angle_)), hop_length=hop_length, win_length=window_length)
                out = gri_lim_1(aaaa, mix_audio, hp.data.nfft, hop_length, window_length)
                score_pesq_gl = pesq(clean_audio, out, 16000)
                score_stoi_gl = stoi(clean_audio, out, 16000)
                score_pesq = pesq(clean_audio, aaa, 16000)
                score_stoi = stoi(clean_audio, aaa, 16000)
                score__5_pesq = score__5_pesq + score_pesq
                score__5_stoi = score__5_stoi + score_stoi
                score__5_pesq_gl = score__5_pesq_gl + score_pesq_gl
                score__5_stoi_gl = score__5_stoi_gl + score_stoi_gl

                print('n__5', n__5, 'score__5_pesq:', score__5_pesq, 'val_pesq:', score__5_pesq / n__5)
                print('n__5', n__5, 'score__5_stoi:', score__5_stoi, 'val_stoi:', score__5_stoi / n__5)
                print('n__5', n__5, 'score__5_pesq_gl:', score__5_pesq_gl, 'val_pesq_gl:', score__5_pesq_gl / n__5)
                print('n__5', n__5, 'score__5_stoi_gl:', score__5_stoi_gl, 'val_stoi_gl:', score__5_stoi_gl / n__5)
            else:
                mix_file_name = "%s.npy" % item_name
                mix_file = np.load(os.path.join(mix_audio_path, mix_file_name))
                n_5 = n_5 + 1
                mix_audio_tf = mix_file[0]
                mix_audio = mix_file[3]
                clean_audio = mix_file[4]
                mix_audio_tf = torch.Tensor(mix_audio_tf)
                mix_tf_0 = librosa.stft(mix_audio, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
                mix_audio_tf_1 = mix_audio_tf.unsqueeze(0)
                mix_audio_tf_1 = mix_audio_tf_1.unsqueeze(0)
                mask, _ = voicefilter_net(mix_audio_tf_1)
                de_tf = mix_audio_tf * mask
                aaaa = torch.squeeze(de_tf)
                aaaa = aaaa.transpose(0, 1).detach().numpy()
                angle_ = np.angle(mix_tf_0)
                aaa = librosa.istft(aaaa * (np.exp(1j * angle_)), hop_length=hop_length, win_length=window_length)
                out = gri_lim_1(aaaa, mix_audio, hp.data.nfft, hop_length, window_length)
                score_pesq_gl = pesq(clean_audio, out, 16000)
                score_stoi_gl = stoi(clean_audio, out, 16000)
                score_pesq = pesq(clean_audio, aaa, 16000)
                score_stoi = stoi(clean_audio, aaa, 16000)
                score_5_pesq = score_5_pesq + score_pesq
                score_5_stoi = score_5_stoi + score_stoi
                score_5_pesq_gl = score_5_pesq_gl + score_pesq_gl
                score_5_stoi_gl = score_5_stoi_gl + score_stoi_gl

                print('n_5', n_5, 'score_5_pesq:', score_5_pesq, 'val_pesq:', score_5_pesq / n_5)
                print('n_5', n_5, 'score_5_stoi:', score_5_stoi, 'val_stoi:', score_5_stoi / n_5)
                print('n_5', n_5, 'score_5_pesq_gl:', score_5_pesq_gl, 'val_pesq_gl:', score_5_pesq_gl / n_5)
                print('n_5', n_5, 'score_5_stoi_gl:', score_5_stoi_gl, 'val_stoi_gl:', score_5_stoi_gl / n_5)
