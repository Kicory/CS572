import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from sklearn.model_selection import train_test_split
import numpy as np
import os


data_path = 'Data_TEST' # add path

hidden_size = 128
num_epochs = 2
input_size = 9
sequence_length = 100
batch_size = 1
learning_rate = 0.001
num_layers = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # x -> batch_size, seq, input_size
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        # out : batch_size, seq_length, hidden_size
        # out (N, 9, 128)
        out = out[:, -1, :]
        out = self.fc(out)


# Data importing and processing 
for path_idx in os.listdir(data_path):
    cur_path = os.path.join(data_path, path_idx)
    print(cur_path)
    input_file = np.loadtxt(cur_path, dtype='float', delimiter=',')
    print(input_file)
    # cell = nn.RNN(input_size=4, hidden_size=2, batchfirst=True)
    inputs = torch.Tensor(input_file)
    print("input size", inputs.size())



lstm = nn.LSTM(9, 1)




# print(device)

#  if torch.rand(1, 1) and torch.cuda.is_available():
#     print("OK")

# print ("Got this working at least")

