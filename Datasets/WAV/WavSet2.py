import matplotlib.pyplot as plt

import os
import torch

import torchaudio


from torch.utils.data import Dataset

import  Datasets.WAV.FastResampler as FastResampler
import myutils

Cache_Dir_Default = "DATA/WAV/CACHE/"
class WavSet(Dataset):


    def __init__(self,
                 filelist,
                 input_size = 64,
                 batch_size = 128,
                 search_depth = -1,
                 samplerate = 8644, stereo = True):
        self.data = []

        self.filelist = myutils.find_files_by_extension(filelist,"wav", search_depth)
        self.samplerate = samplerate

        self.batch_size = batch_size
        self.input_size = input_size
        self.stereo = stereo
        self.noisebuffer = torch.zeros((1,1,1))
        self.cache_dir = Cache_Dir_Default + str(self.samplerate)

        # must be last, LOAD_FILES()!!!
        self.Load_Files()


    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        if self.data is None:
            return 0
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a sample from the dataset at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: The 3D tensor representing the image at the given index.
        """
        sample = self.data[index]

        return sample

    def Randomize(self, grade = 1):

        return

    def Save_cache(self, quiet = False):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        for i, file in enumerate(self.filelist):
            if i >= len(self.data):
                if not quiet:
                    print("Done, no more data to save!")
                break
            ptfile = os.path.basename(file)
            ptfile = os.path.splitext(ptfile)[0]
            ptfile = ptfile + ".pt"
            ptfile = os.path.join(self.cache_dir,ptfile)
            if not os.path.exists(ptfile):
                torch.save(self.data[i],ptfile)
                if not quiet:
                    print("saved {}  out of {} files, name: {}".format(i,len(self.filelist),file))
            else:
                if not quiet:
                    print("skipped file {}  out of {} files, name: {}".format(i,len(self.filelist),file))
    def Load_cache(self, filelist, quiet = False):
        uncached_list = []
        for file in filelist:

            ptfile = os.path.basename(file)
            ptfile = os.path.splitext(ptfile)[0]
            ptfile = ptfile + ".pt"
            ptfile = os.path.join(self.cache_dir,ptfile)
            if os.path.exists(ptfile):
                if not quiet:
                    print("loaded from cache {} : {}".format(self.samplerate,file))

                self.data.append(torch.load(ptfile))

            else:
                uncached_list.append(file)

        for file in uncached_list:
            self.filelist.pop(self.filelist.index(file))
            self.filelist.append(file)

        return uncached_list

    def to(self, device):
        for i in range(len(self.data)):
            self.data[i] = self.data[i].to(device)
        return self


    def Load_Files(self):
        self.data = []
        resampler = None
        sepct = torchaudio.transforms.Spectrogram()
        rin = None
        rout = None
        uncached_list = self.Load_cache(self.filelist)
        cached_num = len(self.data) - len(uncached_list)
        for i,file in enumerate(uncached_list):
            waveform, current_sample_rate = torchaudio.load(file, normalize=True)

            if current_sample_rate != self.samplerate:

                if resampler is None or \
                rin != current_sample_rate or \
                rout != self.samplerate:
                    rin = current_sample_rate
                    rout = self.samplerate
                    #resampler = torchaudio.transforms.Resample(rin, rout)
                    resampler = FastResampler.FastResampler(rin, rout,dtype=waveform.dtype)

                #resample
                waveform = (resampler(waveform))

            self.data.append(sepct(torch.mean(myutils.normalize_volume(waveform), dim=0).unsqueeze(0)).transpose(1,2))


            print("loaded {}  out of {} files, name: {}".format(i + cached_num,len(self.filelist),file))

            self.Save_cache(quiet=True)
        return





