import torch
from torch import nn
import numpy as np


class GRUCellV2(nn.Module):
    """
    GRU cell implementation
    """
    def __init__(self, input_size, hidden_size, activation=torch.tanh):
        """
        Initializes a GRU cell

        :param      input_size:      The size of the input layer
        :type       input_size:      int
        :param      hidden_size:     The size of the hidden layer
        :type       hidden_size:     int
        :param      activation:      The activation function for a new gate
        :type       activation:      callable
        """
        super(GRUCellV2, self).__init__()
        self.activation = activation

        # initialize weights by sampling from a uniform distribution between -K and K
        K = 1 / np.sqrt(hidden_size)
        # weights
        self.w_ih = nn.Parameter(torch.rand(3 * hidden_size, input_size) * 2 * K - K)
        self.w_hh = nn.Parameter(torch.rand(3 * hidden_size, hidden_size) * 2 * K - K)
        self.b_ih = nn.Parameter(torch.rand(3 * hidden_size) * 2 * K - K)
        self.b_hh = nn.Parameter(torch.rand(3 * hidden_size) * 2 * K - K)

        
    def forward(self, x, h):
        """
        Performs a forward pass through a GRU cell


        Returns the current hidden state h_t for every datapoint in batch.
        
        :param      x:    an element x_t in a sequence
        :type       x:    torch.Tensor
        :param      h:    previous hidden state h_{t-1}
        :type       h:    torch.Tensor
        """
        #
        # YOUR CODE HERE
        #
        wi_x = torch.chunk(torch.matmul(self.w_ih, x), 3)
        wh_h = torch.chunk(torch.matmul(self.w_hh, h), 3)
        b_ih = torch.chunk(self.b_ih, 3)
        b_hh = torch.chunk(self.b_hh, 3)

        rt = torch.sigmoid(wi_x[0] + self.flat_transpose(b_ih[0]) + wh_h[0] + self.flat_transpose(b_hh[0]))  # reset gate
        zt = torch.sigmoid(wi_x[1] + self.flat_transpose(b_ih[1]) + wh_h[1] + self.flat_transpose(b_hh[1]))  # update gate
        nt = torch.tanh(wi_x[2] + self.flat_transpose(b_ih[2]) + rt * (wh_h[2] + self.flat_transpose(b_hh[2]))) # tentative new hidden state
        ht = (1 - zt) * nt + zt * h  # new hidden state
        return ht

    def flat_transpose(self, tensor):
        return torch.reshape(tensor, (tensor.size()[0], 1))


class GRU2(nn.Module):
    """
    GRU network implementation
    """
    def __init__(self, input_size, hidden_size, bias=True, activation=torch.tanh, bidirectional=False):
        super(GRU2, self).__init__()
        self.bidirectional = bidirectional
        self.fw = GRUCellV2(input_size, hidden_size, activation=activation) # forward cell
        if self.bidirectional:
            self.bw = GRUCellV2(input_size, hidden_size, activation=activation) # backward cell
        self.hidden_size = hidden_size
        
    def forward(self, x):
        """
        Performs a forward pass through the whole GRU network, consisting of a number of GRU cells.
        Takes as input a 3D tensor `x` of dimensionality (B, T, D),
        where B is the batch size;
              T is the sequence length (if sequences have different lengths, they should be padded before being inputted to forward)
              D is the dimensionality of each element in the sequence, e.g. word vector dimensionality

        The method returns a 3-tuple of (outputs, h_fw, h_bw), if self.bidirectional is True,
                           a 2-tuple of (outputs, h_fw), otherwise
        `outputs` is a tensor containing the output features h_t for each t in each sequence (the same as in PyTorch native GRU class);
                  NOTE: if bidirectional is True, then it should contain a concatenation of hidden states of forward and backward cells for each sequence element.
        `h_fw` is the last hidden state of the forward cell for each sequence, i.e. when t = length of the sequence;
        `h_bw` is the last hidden state of the backward cell for each sequence, i.e. when t = 0 (because the backward cell processes a sequence backwards)
        
        :param      x:    a batch of sequences of dimensionality (B, T, D)
        :type       x:    torch.Tensor
        """
        #
        # YOUR CODE HERE
        #

        B, T, D = x.size()
        h_fw2 = torch.zeros(self.hidden_size, B)
        outputs_fw = torch.zeros(B, T, self.hidden_size)
        for t in range(T):
            xt = torch.transpose(x[:, t, :], 0, 1)
            h_fw2 = self.fw(xt, h_fw2)
            outputs_fw[:, t, :] = torch.transpose(h_fw2, 0, 1)

        if self.bidirectional:
            outputs_bw = torch.zeros(B, T, self.hidden_size)
            h_bw2 = torch.zeros(self.hidden_size, B)
            for t in range(T - 1, -1, -1):
                xt = torch.transpose(x[:, t, :], 0, 1)
                h_bw2 = self.bw(xt, h_bw2)
                outputs_bw[:, t, :] = torch.transpose(h_bw2, 0, 1)
            outputs = torch.cat((outputs_fw, outputs_bw), dim=2)
            return outputs, torch.transpose(h_fw2, 0, 1), torch.transpose(h_bw2, 0, 1)

        return outputs_fw, torch.transpose(h_fw2, 0, 1)



def is_identical(a, b):
    return "Yes" if np.all(np.abs(a - b) < 1e-6) else "No"


if __name__ == '__main__':
    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru = nn.GRU(10, 20, bidirectional=False, batch_first=True)
    outputs, h = gru(x)

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru2 = GRU2(10, 20, bidirectional=False)
    outputs, h_fw = gru2(x)
    
    print("Checking the unidirectional GRU implementation")
    print("Same hidden states of the forward cell?\t\t{}".format(
        is_identical(h[0].detach().numpy(), h_fw.detach().numpy())
    ))

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru = GRU2(10, 20, bidirectional=True)
    outputs, h_fw, h_bw = gru(x)

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru2 = nn.GRU(10, 20, bidirectional=True, batch_first=True)
    outputs, h = gru2(x)
    
    print("Checking the bidirectional GRU implementation")
    print("Same hidden states of the forward cell?\t\t{}".format(
        is_identical(h[0].detach().numpy(), h_fw.detach().numpy())
    ))
    print("Same hidden states of the backward cell?\t{}".format(
        is_identical(h[1].detach().numpy(), h_bw.detach().numpy())
    ))