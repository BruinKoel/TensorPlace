import torch, torchaudio
import Datasets.WAV.FastResampler as FastResampler
from datetime import datetime
import os
import random

def guided_inference(model,initial_input,length,input_sample_rate,output_file = "",sequence_length = 256):

    model.eval()
    random_range = [ random.randint(0,initial_input.shape[1]-1)]
    random_range.append(random_range[0] + random.randint(random_range[0],initial_input.shape[1]-1)//4)
    output = model.autoregressive_inference(length,initial_input[:,random_range[0]:random_range[1],:],volume=1.0)

    sampler = FastResampler.FastResampler(input_sample_rate,44100,dtype=output.dtype)
    waveform = sampler(torch.flatten(output.cpu()).unsqueeze(0))
    if output_file == "":
        output_file = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "Inference.wav"
        output_file = os.path.join("INFERENCEOUT",output_file)
    torchaudio.save(output_file,waveform,44100)
    model.train()