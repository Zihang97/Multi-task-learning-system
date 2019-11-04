import torch
import numpy as np
import os
import glob
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import copy
from random import sample, shuffle
import torch.nn.functional as F
import librosa
from torch.utils.data import DataLoader, Dataset
from hparam import hparam as hp
import time

gpu_dtype = torch.cuda.FloatTensor

layers = 4          # number of LSTM layers
hidden_size = 512   # number of cells per direction
num_dirs = 2        # number of LSTM directions

assert num_dirs == 1 or num_dirs == 2, 'num_dirs must be 1 or 2'

def _collate_fn(batch):

   batch = sorted(batch, reverse = True, key = lambda p: p[0].shape[0])
   longest_sample = batch[0][0]
   feat_size = longest_sample.shape[1]
   minibatch_size = len(batch)
   max_seqlength = longest_sample.shape[0]
   inputs = torch.zeros(minibatch_size, max_seqlength, feat_size)
   clean = torch.zeros(minibatch_size, max_seqlength, feat_size)
   input_sizes =  torch.IntTensor(minibatch_size)
   target_sizes = torch.IntTensor(minibatch_size)
   targets = []
   input_sizes_list = []
   for x in range(minibatch_size):
        sample = batch[x]
        tensor = torch.Tensor(sample[0])
        target = sample[1]
        clean_1 = torch.Tensor(sample[2])
        seq_length = tensor.shape[0]
        inputs[x].narrow(0, 0, seq_length).copy_(tensor)
        clean[x].narrow(0, 0, seq_length).copy_(clean_1)
        input_sizes[x] = seq_length
        input_sizes_list.append(seq_length)
        target_sizes[x] = len(target)
        targets.extend(target)
   targets = torch.IntTensor(targets)
   inputs = torch.Tensor(inputs)
   clean = torch.Tensor(clean)
   return inputs, targets, input_sizes, target_sizes, clean

class dataset_preprocess_voicefilter(Dataset):
    def __init__(self, shuffle=False, utter_start=0):

        self.file_path = '/raid/data/lps/data_ctc/train'
        self.file_list = os.listdir(self.file_path)
        self.shuffle = shuffle
        self.utter_start = utter_start
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, item):
        np_file_list = os.listdir(self.file_path)

        selected_file = np_file_list[item]
        utters_file = np.load(os.path.join('/raid/data/lps/data_ctc/train', selected_file))

        return utters_file


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

def train():
    dset = dataset_preprocess_voicefilter()
    loader = DataLoader(dset, batch_size=512, shuffle=True, collate_fn=_collate_fn, num_workers=10, pin_memory=False)
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    lr_factor = 4
    iteration = 0
    voicefilter_net = VoiceFilter().cuda(device_ids[0])
    voicefilter_net = torch.nn.DataParallel(voicefilter_net, device_ids=device_ids)
    vf_loss = nn.MSELoss()
    optimizer_vf = torch.optim.Adam(voicefilter_net.parameters(), lr=0.0001*lr_factor)
    voicefilter_net.train()
    for e in range(1000):
        total_loss = 0
        total_loss_2 = 0
        total_loss_1 = 0
        for num, (inputs, targets, input_sizes, target_sizes, clean) in enumerate(loader):
            inputs = inputs.cuda(device_ids[0])
            clean_gpu = clean.cuda(device_ids[0])
            optimizer_vf.zero_grad()
            inputs_1 = inputs.unsqueeze(1)
            mask, output_ctc = voicefilter_net(inputs_1)
            de_tf =inputs * mask
            output_ctc = torch.transpose(output_ctc, 0, 1)
            output_ctc = output_ctc.log_softmax(2)
            output_ctc = output_ctc.cpu()
            loss_1 = F.ctc_loss(output_ctc, targets, input_sizes, target_sizes)
            loss_1 = loss_1.cuda(device_ids[0])
            loss_2 = vf_loss(de_tf, clean_gpu)
            loss = loss_1 + loss_2
            loss.backward()
            optimizer_vf.step()
            total_loss = total_loss + loss
            total_loss_1 = total_loss_1 + loss_1
            total_loss_2 = total_loss_2 + loss_2
            iteration = iteration + 1
            if (iteration % 100 == 0):
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tLoss_1:{6:.4f}\tLoss_2:{7:.4f}\tTotal_loss_ave:{8:.6f}\t\n".format(
                    time.ctime(), e + 1, num, len(dset) // (128), iteration, loss, loss_1, loss_2, total_loss / num)
                print(mesg)
        mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTotal_Loss_1:{6:.4f}\tTotal_Loss_2:{7:.4f}\tTotal_loss_ave:{8:.6f}\t\n".format(
            time.ctime(), e + 1, num, len(dset) // (128), iteration, loss, total_loss_1/num, total_loss_2/num,
                          total_loss / num)
        print(mesg)
        if(e + 2) % 1 == 0:
            voicefilter_net.eval()
            voicefilter_net.cpu()
            model_mid_name = 'epoch_' + str(e + 1) + ".pth"
            model_mid_path = os.path.join('/raid/data/lps/model/ctc_1/', model_mid_name)
            torch.save(voicefilter_net.state_dict(), model_mid_path)
            voicefilter_net.cuda(device_ids[0]).train()
        if (e + 1) % 10 == 0:
            lr_factor = lr_factor / 2
            print("iteration:", iteration)


if __name__ == '__main__':
    train()