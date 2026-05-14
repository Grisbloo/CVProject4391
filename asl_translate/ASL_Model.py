"""
ASL_Model.py
TL:DR This program serves as the model for which the entire backbone of the ASL program runs off of. An LSTM (Long Short-Term Memory) receives the batches of .npy files and classes them down per video of letter.
It contains a 30% drop-out system is also implemented so it does not memorize the data.
"""

import torch
import torch.nn as nn

class ASLSequenceInterpreter(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2, num_classes=26):
        super(ASLSequenceInterpreter, self).__init__()
        
        # THE MEMORY LAYER (LSTM) 
        # input_size = 63 (because 21 MediaPipe joints * 3 coordinates (x,y,z))
        # hidden_size = 64 (the number of "neurons" parsing the patterns)
        # batch_first=True tells PyTorch data is structured as (Batch, Sequence, Features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # THE DECISION LAYER
        # Processes the sequence, and force the model to make a guess
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU() # Activation function (keeps the math non-linear)
        self.dropout = nn.Dropout(0.3) # Prevents the LSTM from just memorizing the data
        
       # 64 internal features squashed down to the 26 classes
        self.fc2 = nn.Linear(64, num_classes) 

    def forward(self, x):
        # x is input data coming in. Shape: (batch_size, 16 frames, 63 points)
        
        # Pass the 16 frames through the LSTM
        out, (hn, cn) = self.lstm(x)
        
        # The LSTM gives us an output for ALL 16 frames. 
        # Only the network's final conclusion matters so slice it out after seeing the 16th frame.
        out = out[:, -1, :] 
        
        # Pass the conclusion through the decision layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Returns an array of probabilities for each letter
        return out
    
