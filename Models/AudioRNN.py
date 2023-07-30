import torch
import torch.nn as nn
import torch_directml


from Datasets.WAV.SpectrumComposer import SpectrumComposer

class AudioPredictor(nn.Module):
    def __init__(self, input_size = 1024, hidden_size = 512, output_size = 1024, num_layers = 3, dropout=0.2):
        super(AudioPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.gelu = torch.nn.GELU()

        self.linearin = nn.Linear(input_size,hidden_size)

        self.celu = nn.CELU(inplace=True)


        # Define your GRU layers
        self.grulist = nn.ModuleList( [nn.GRU(hidden_size, hidden_size, batch_first=True) for _ in range(num_layers )] )


        self.linear = nn.Linear(hidden_size, output_size)
        self.last_hidden = None


        self.dropout = nn.Dropout(dropout,inplace=False)



    def forward(self, x):
        # Implement the forward pass
        # x: (sequence_length, sequence_length, input_size)

        batch_size, sequence_length, _ = x.size()

        # Initialize hidden states for GRU

        hidden_states = [torch.zeros(1, batch_size, self.hidden_size,device=x.device,requires_grad=False) for _ in range(self.num_layers + 1)]

        # GRU forward pass for each layer

        gru_out = self.linearin(x)

        for layer in range(self.num_layers):

            gru_out, hidden_states[layer] = self.grulist[layer](gru_out, hidden_states[layer])
            gru_out = self.dropout(gru_out)


        # Linear layer to get the predicted waveform for each time step
        output = self.linear(gru_out)

        #del gru_out, hidden_states, h, x
        del gru_out, hidden_states, x

        return output


    def autoregressive_inference(self, n, pre_sequence=None,device = None,volume = None):
        if device is None:
            device = torch.device("cpu")
        # n: Number of steps to generate
        # pre_sequence: Optional sequence preceding the generated output (can be None)
        self.to(device)
        # Initialize the hidden states
        batch_size = 1  # Since we generate one step at a time, set batch_size to 1
        hidden_states = [(torch.zeros(1, batch_size, self.hidden_size,device=device,requires_grad=False))
                         for _ in range(self.num_layers)]

        if pre_sequence is None:
            # If no pre_sequence is provided, generate output from a zero input
            pre_sequence = torch.zeros(1, 1, self.input_size,device=self.device)


        # Generate n steps autoregressively
        generated_sequence = []
        output = None
        def forward_step(x, hidden_states,volume = None):


            gru_out = self.linearin(x)

            for layer in range(self.num_layers):
                gru_out, hidden_states[layer] = self.grulist[layer](gru_out, hidden_states[layer])

            # Linear layer to get the predicted waveform for each time step
            output = self.linear(gru_out)

            # del gru_out, hidden_states, h, x
            if volume is None:
                return output, hidden_states

            return volume * output, hidden_states

        output , hidden_states = forward_step(pre_sequence,hidden_states)
        generated_sequence.append(output)
        for _ in range(n):
            output,hidden_states = forward_step(output[:,output.shape[1]-1:],hidden_states)
            generated_sequence.append(output)


        # Convert the generated_sequence list to a tensor

        return torch.cat((pre_sequence,generated_sequence[0],torch.stack(generated_sequence[1:],dim = 2).squeeze(0)), dim=1)
