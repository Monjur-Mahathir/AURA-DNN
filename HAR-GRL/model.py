import time
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output
        return grad_input, None
revgrad = GradientReversal.apply


class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)
    

class CSIHAR(nn.Module):
    def __init__(self, in_channels, num_class, num_domains=9):
        super(CSIHAR, self).__init__()
        self.in_channels = in_channels
        self.num_class = num_class

        self.norm = nn.LayerNorm(normalized_shape=[self.in_channels, 500, 242])

        self.first_cnn_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 3)),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.ReLU()
        )

        self.second_cnn_block = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(3, 2)),
            nn.ReLU()
        )

        self.third_cnn_block = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(input_size=576, hidden_size=128, num_layers=3, batch_first=True, bidirectional=True)

        self.rev = GradientReversal(alpha=1.)
        self.fc_domain = nn.Linear(in_features=256, out_features=num_domains)

        self.fc = nn.Linear(in_features=256, out_features=num_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.norm(x)
        out = self.first_cnn_block(out)
        out = self.second_cnn_block(out)
        out = self.third_cnn_block(out)
        
        out = out.permute(0, 2, 1, 3)
        out = out.reshape([out.shape[0], out.shape[1], out.shape[2] * out.shape[3]])

        out, _ = self.lstm(out)
        out = out[:, -1, :]
        
        pred = self.fc(out)
        pred = self.softmax(pred)

        domain = self.rev(out)
        domain = self.fc_domain(domain)
        domain = self.softmax(domain)

        return pred, domain
