import torch, torchaudio
import Datasets.WAV.FastResampler as FastResampler
from datetime import datetime
import os
import torch.nn as nn
import math

import random

class SpectrumComposer(nn.Module):

    def __init__(self,size_in,size_out,sample_rate = 8644,max_wave_size = 1024 *1024,device = None):

        assert (size_in % 2 ) == 0
        super().__init__()
        if device is None:
            device = torch.device("cpu")

        self.device = device
        self.wavetable = []
        self.sample_rate = sample_rate

        self.size_in = size_in

        self.size_out = size_out
        self.max_wave_size = max_wave_size

        weights = torch.Tensor([self.sample_rate * (1/i) for i in range(1,(size_in//2)+1)])
        self.weights = torch.nn.Parameter(weights)




    def forward(self, input):
        output = torch.zeros(input.shape[0],self.size_out,input.shape[2],device = self.device)
        batches, batch_size, channels = input.shape
        for batch_num in range(batches):
            for seq_num in range(batch_size):
                sine = torch.stack([torch.linspace(phase.item() , self.weights[i].item() + phase.item() , self.size_out) for i,phase in enumerate(input[batch_num,seq_num,:self.size_in//2])], dim=0)
                output[batch_num,seq_num,:] = torch.sum(torch.sin(sine)*input[batch_num,seq_num,self.size_in//2:].unsqueeze(0).transpose(1,0),dim=0)

        return output



    def generate_wavetables(self):
        self.wavetable = []
        for i in range(len(self.weights)):
            self.wavetabe.append(None);
            wave_span = int(self.weights[i])
            wave_size = wave_span
            wave_count = 1
            while wave_size % self.size_out != 0 and\
                wave_size < self.max_wave_size:
                wave_size += self.size_out
                wave_count += 1

            self.wavetable[i] = torch.sin(torch.linspace(0, 2 * math.pi, wave_span))
            self.wavetable[i] = self.wavetable[i].view(1,-1,self.size_out)
            #double the table to allow for phase shifting
            self.wavetable[i] = torch.cat((self.wavetable[i][:,1:,:],self.wavetable[i][:,:-1,:]),dim=2)



    def get_workable_lcm(self,degree:float):
        i = 0
        space = int(degree * self.size_out)
        max_lcm_size = self.size_out * self.size_out
        lcm = max_lcm_size +1
        while lcm > max_lcm_size:


            lcm = torch.lcm(space + i, self.size_out)
            i -= i + 1
            space
        return lcm





