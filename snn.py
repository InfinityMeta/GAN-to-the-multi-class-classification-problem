import torch.nn as nn

class ShallowNN(nn.Module):
    def __init__(self):
        super(ShallowNN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=6, stride=6, bias=False),
            nn.Linear(in_features=20, out_features=20),
            nn.Softmax(dim=2)
        )

    def forward(self, input):
        return self.main(input)