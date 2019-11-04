#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:56:19 2018

@author: harry
"""
import librosa
import numpy as np
#import torch
# import torch.autograd as grad
#import torch.nn.functional as F

from hparam import hparam as hp

def get_centroids(embeddings):
    centroids = []
    for speaker in embeddings:
        centroid = 0
        for utterance in speaker:
            centroid = centroid + utterance
        centroid = centroid/len(speaker)
        centroids.append(centroid)
    # stack拼接会增加维度，默认应该�?维度增加
    centroids = torch.stack(centroids)
    return centroids

def get_centroid(embeddings, speaker_num, utterance_num):
    centroid = 0
    for utterance_id, utterance in enumerate(embeddings[speaker_num]):
        if utterance_id == utterance_num:
            continue
        centroid = centroid + utterance
    centroid = centroid/(len(embeddings[speaker_num])-1)
    return centroid

def gri_lim_1(stft_amg, mix, n_fft, hop_length, win_length):
    shape = (stft_amg.shape[1]-1) * hop_length
    x = mix
    for i in range(100):
        x_stft = librosa.stft(x, n_fft, hop_length, win_length)
        angle_ = np.angle(x_stft)
        x = librosa.istft(stft_amg * (np.exp(1j*angle_)), hop_length, win_length)
    return x
    
def gri_lim(stft_amg, n_fft, hop_length, win_length):
    shape = (stft_amg.shape[1]-1) * hop_length
    x = np.random.rand(shape)
    for i in range(max_iter):
        x_stft = librosa.stft(x, n_fft, hop_length, win_length)
        x = librosa.istft(stft_amg * (x_stft/np.abs(x_stft)), hop_length, win_length)
    return x
    
def get_cossim(embeddings, centroids):
    # Calculates cosine similarity matrix. Requires (N, M, feature) input
    cossim = torch.zeros(embeddings.size(0),embeddings.size(1),centroids.size(0))
    for speaker_num, speaker in enumerate(embeddings):
        for utterance_num, utterance in enumerate(speaker):
            for centroid_num, centroid in enumerate(centroids):
                if speaker_num == centroid_num:
                    centroid = get_centroid(embeddings, speaker_num, utterance_num)
                ####这位什�?1e-6?
                output = F.cosine_similarity(utterance,centroid,dim=0)+1e-6
                cossim[speaker_num][utterance_num][centroid_num] = output
    return cossim

def calc_loss(sim_matrix):
    # Calculates loss from (N, M, K) similarity matrix
    per_embedding_loss = torch.zeros(sim_matrix.size(0), sim_matrix.size(1))
    for j in range(len(sim_matrix)):
        for i in range(sim_matrix.size(1)):
        #  这也�?e-6
            per_embedding_loss[j][i] = -(sim_matrix[j][i][j] - ((torch.exp(sim_matrix[j][i]).sum()+1e-6).log_()))
    loss = per_embedding_loss.sum()    
    return loss, per_embedding_loss

def normalize_0_1(values, max_value, min_value):
    normalized = np.clip((values - min_value) / (max_value - min_value), 0, 1)
    return normalized

def mfccs_and_spec(wav_file, wav_process = False, calc_mfccs=False, calc_mag_db=False):    
    sound_file, _ = librosa.core.load(wav_file, sr=hp.data.sr)
    window_length = int(hp.data.window*hp.data.sr)
    hop_length = int(hp.data.hop*hp.data.sr)
    duration = hp.data.tisv_frame * hp.data.hop + hp.data.window
    
    # Cut silence and fix length
    if wav_process == True:
        sound_file, index = librosa.effects.trim(sound_file, frame_length=window_length, hop_length=hop_length)
        length = int(hp.data.sr * duration)
        sound_file = librosa.util.fix_length(sound_file, length)
        
    spec = librosa.stft(sound_file, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
    mag_spec = np.abs(spec)
    
    mel_basis = librosa.filters.mel(hp.data.sr, hp.data.nfft, n_mels=hp.data.nmels)
    mel_spec = np.dot(mel_basis, mag_spec)
    
    mag_db = librosa.amplitude_to_db(mag_spec)
    #db mel spectrogram
    mel_db = librosa.amplitude_to_db(mel_spec).T
    
    mfccs = None
    if calc_mfccs:
        mfccs = np.dot(librosa.filters.dct(40, mel_db.shape[0]), mel_db).T
    
    return mfccs, mel_db, mag_db


def for_stft(audio_path):
    window_length = int(hp.data.window_gan * hp.data.sr)
    hop_length = int(hp.data.hop * hp.data.sr)
    audio, _ = librosa.load(audio_path, sr=hp.data.sr)
    audio_t = librosa.stft(audio, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
    audio_tf = np.abs(audio_t)
    return audio_tf
def for_stft_1(audio_path):
    window_length = int(hp.data.window * hp.data.sr)
    hop_length = int(hp.data.hop * hp.data.sr)
    audio, _ = librosa.load(audio_path, sr=hp.data.sr)
    audio_t = librosa.stft(audio, n_fft=320, hop_length=160, win_length=320)
    audio_tf = np.abs(audio_t)
    return audio_tf
def for_stft_2(audio):
    window_length = int(hp.data.window_gan * hp.data.sr)
    hop_length = int(hp.data.hop * hp.data.sr)
    # audio, _ = librosa.load(audio_path, sr=hp.data.sr)
    audio_t = librosa.stft(audio, n_fft=hp.data.nfft, hop_length=hop_length, win_length=window_length)
    audio_tf = np.abs(audio_t)
    return audio_tf
# if __name__ == "__main__":
#     w = grad.Variable(torch.Tensor(1.0))
#     b = grad.Variable(torch.Tensor(0.0))
#     embeddings = torch.Tensor([[0,1,0],[0,0,1], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]).to(torch.float).reshape(3,2,3)
#     centroids = get_centroids(embeddings)
#     cossim = get_cossim(embeddings, centroids)
#     sim_matrix = w*cossim + b
#     loss, per_embedding_loss = calc_loss(sim_matrix)