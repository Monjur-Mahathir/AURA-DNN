import time
import torch
import torch.nn as nn


class CSIHAR(nn.Module):
    def __init__(self, in_channels, num_class):
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
        
        out = self.fc(out)
        out = self.softmax(out)
        return out


if __name__ == "__main__":
    channels = 1
    num_classes = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = CSIHAR(in_channels=channels, num_class=num_classes)
    model = model.to(device)

    example_input = torch.zeros(1, channels, 500, 242)
    example_input = example_input.to(device)

    t0 = time.time()
    out = model(example_input)
    t1 = time.time()

    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_count += param.nelement()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    print('parameters: {:.3f}'.format(param_count))
    print(model)