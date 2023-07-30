from cProfile import Profile
from pstats import Stats, SortKey
import math
import random

import torch
import multiprocessing
import numpy as np

import os
import time

import matplotlib.pyplot as plt

import Inference.AudioInfer as AudioInfer

import torch_directml as torch_directml

from Models.AudioRNN import AudioPredictor
from  Datasets.WAV.WavSet import WavSet
sequence_length = 2048
torch.set_num_threads(16)

class AudioHost:
    default_model_path = "audiornn.pth"
    def __init__(self, model_path = default_model_path,
                 dataset = None,
                 input_size = 128,
                 hidden_size = 64,
                 num_layers = 4,
                 dropout=0.1):

        #self.model_path = model_path
        #self.device = torch_directml.device()
        self.device = torch.device("cpu")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropoutlayer1 = torch.nn.Dropout(0.001)

        self.load_model(model_path)

        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=0.06)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=20, factor=0.60, verbose=True)
        print(self.optimizer)
        self.dataset  = dataset
        print(self.model)

        self.last_step_loss = 0
        self.max_dim0 = 1024

        self.training_process = None  # New process variable instead of thread
        self._stop_training = multiprocessing.Event()  # Use multiprocessing.Event() for process-safe signaling

    def criterion(self, output, target,window_size = 3):
        mse = torch.nn.MSELoss()

        x = output
        y = target
        i = -window_size
        losses = []
        while i < window_size:

            comp = None
            compout = None
            if i < 0:
                comp = x[:,:,:x.shape[2] + i]
                compout = y[:,:,-i:]
            elif i > 0:
                comp = x[:,:, i:]
                compout = y[:,:,:y.shape[2] -i]
            else:
                comp = x
                compout = y
            losses.append(mse(comp, compout))
            i += 1

        if window_size == 0:
            return mse(output, target)

        return min(losses)
    def load_model(self, model_path = default_model_path):

        if model_path is not None:
            self.model_path = model_path
        if not os.path.exists(model_path):
            print("Model file does not exist but location is saved.")

            self.model = AudioPredictor(self.input_size, self.hidden_size,
                        self.input_size, self.num_layers,self.dropout).to(self.device)
            return

        # Load the model from the provided .pth file
        print("Loading model from " + self.model_path)
        self.model = torch.load(self.model_path)


    def run_inference(self):
        i = random.randint(0,len(self.dataset)-1)
        AudioInfer.guided_inference(self.model,self.dataset[i],1000,self.dataset.samplerate,sequence_length=sequence_length)


    def save_model(self, model_path = default_model_path):
        if model_path is not None:
            self.model_path = model_path
        # Save the model to the provided .pth file
        print("Saving model to " + self.model_path)
        torch.save(self.model, self.model_path)
        self.run_inference()

    def train(self, num_epochs=8000, dataset=None, dataset_path=None):
        if dataset is not None:
            self.dataset = dataset
        elif dataset_path is not None and self.dataset is None:
            self.dataset = WavSet(dataset_path, batch_size=sequence_length, input_size=self.model.input_size,)
        else:
            print("No dataset provided.")
            return

        self.loss_history = []

        # Start the training process in a new process
        self._train_thread(num_epochs, sequence_length, self.dataset)
        #self.training_process = multiprocessing.Process(target=self._train_thread,
         #                                               args=(num_epochs, sequence_length, self.dataset))
        #self.training_process.start()

    save_time = 4*60
    max_data_length = 1024*1024*1024#~4GB
    def _train_thread(self, num_epochs, sequence_length, dataset):


        self.model.to(self.device)



        next_save = time.time() + self.save_time
        average_loss = -1
        index = [x for x in range(len(dataset))]

        for epoch in range(num_epochs):
            i = 0

            #with Profile() as profile:
            random.shuffle(index)
            self.model.train()
            while i < len(dataset):



                item_time = time.time()

                if time.time() > next_save:

                    self.save_model()
                    self.model.to(self.device)
                    next_save = time.time() + self.save_time
                    self.model.train()


                if self._stop_training.is_set():
                    print("Training aborted.")
                    return
                # Training steps...
                apgrad = 0
                length = 9
                if i % length == length-1:
                    apgrad = length
                speed = self.training_step(index[i], dataset,apply_grad = apgrad)

                i += 1

                average_loss = sum(self.loss_history) / len(self.loss_history)
                print("Completed file {} out of {} with {} Sec/s average loss: {}".format(i,len(dataset),speed,average_loss))
                #print("{}".format(Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats()))




            print(f"Epoch {epoch + 1} completed.")


    def training_step(self,i, dataset,apply_grad):
        data_item = dataset[i].to(self.device)
        mse = torch.nn.MSELoss().to(self.device)

        #data_item = self.dropoutlayer1(data_item)


        item_time = time.time()
        sequence_start = 0
        num_batches = (data_item.shape[1] - sequence_start - 1) // sequence_length


        space = torch.cat(([
            data_item
            [:, sequence_start + j * sequence_length:sequence_start + (j + 1) * sequence_length + 1, :]
            .view(1, sequence_length + 1, -1)
            for j in range(min(num_batches,self.max_dim0))
        ]), dim=0)



        input = space[:, :-1, :]

        target = space[:, 1:, :]
        numell = space.numel()



        if self._stop_training.is_set():
            print("Training aborted.")
            return
        # Training steps...
        loss = []
        tries = 1


        stack = torch.cat([(torch.rand(input.shape, device=input.device) * 0.0001) + input for _ in range(tries)], dim=0)
        output = self.model(stack)
        loss = mse(output, target.repeat(tries,1,1))
        self.loss_history.append(loss.item())

        loss.backward()

        if apply_grad > 0:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = param.grad / apply_grad

            self.optimizer.step()
            self.loss_history =   self.loss_history[max(len(self.loss_history)-len(dataset)*10, 0):]
            average_loss = sum(self.loss_history) / len(self.loss_history)
            print("step-grad loss: {} change: {}".format(average_loss, average_loss - self.last_step_loss))
            self.last_step_loss = average_loss
            self.scheduler.step(average_loss)
            self.optimizer.zero_grad()

        print(f"Loss: {loss.item()}")

        sequence_start += sequence_length
        i += 1
        speed = numell / (time.time() - item_time) / self.dataset.samplerate
        #print("Completed file {} out of {} with {} Sec/s".format(i, len(dataset), speed))

        return speed



    def abort_training(self):
        # Signal the training thread to stop
        self._stop_training.set()

    def inference(self, input_data):
        # Run inference on the loaded model
        if self.model is None:
            print("Model has not been loaded.")
            return

        with torch.no_grad():
            output = self.model(input_data)
            return output

# Example usage:
if __name__ == "__main__":
    pass

