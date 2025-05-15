import torch
import torchvision
import torch.nn as nn

## classes defined in here:
# ------------------------
# PetsConvNetBaseline - this was defined in the question1.ipynb
# PetsConvNetNoBatchNorm - baseline without batch normalization
# PetsConvNetLeakyRelu - baseline with a leakyReLU activation function
# PetsConvNetGelu - baseline with a GELU activation function
# PetsConvNetL5 - baseline with 5 layers
# PetsConvNetL7 - baseline with 7 layers


# the baseline (basic) PetsConvNet CNN
class PetsConvNetBaseline(nn.Module):
    def __init__(self, num_classes=4):
        super(PetsConvNetBaseline, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l1 output dim = 128 - 5 + 1 + 2*2 / 2
        #               = 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l2 output dim = 64 - 5 + 1 + 2*2 / 2
        #               = 32

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l3 output dim = 32 - 5 + 1 + 2*2 / 2
        #               = 16

        self.fc = nn.Linear(16*16*64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# Baseline except without batch normalization
class PetsConvNetNoBatchNorm(nn.Module):
    def __init__(self, num_classes=4):
        super(PetsConvNetNoBatchNorm, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l1 output dim = 128 - 5 + 1 + 2*2 / 2
        #               = 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l2 output dim = 64 - 5 + 1 + 2*2 / 2
        #               = 32

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l3 output dim = 32 - 5 + 1 + 2*2 / 2
        #               = 16

        self.fc = nn.Linear(16*16*64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# Baseline except with LeakyRelu
class PetsConvNetLeakyRelu(nn.Module):
    def __init__(self, num_classes=4):
        super(PetsConvNetLeakyRelu, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l1 output dim = 128 - 5 + 1 + 2*2 / 2
        #               = 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l2 output dim = 64 - 5 + 1 + 2*2 / 2
        #               = 32

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l3 output dim = 32 - 5 + 1 + 2*2 / 2
        #               = 16

        self.fc = nn.Linear(16*16*64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# baseline except with GELU
class PetsConvNetGelu(nn.Module):
    def __init__(self, num_classes=4):
        super(PetsConvNetGelu, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l1 output dim = 128 - 5 + 1 + 2*2 / 2
        #               = 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l2 output dim = 64 - 5 + 1 + 2*2 / 2
        #               = 32

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l3 output dim = 32 - 5 + 1 + 2*2 / 2
        #               = 16

        self.fc = nn.Linear(16*16*64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# Baseline with 5 Layers
class PetsConvNetL5(nn.Module):
    def __init__(self, num_classes=4):
        super(PetsConvNetL5, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l1 output dim = 128 - 5 + 1 + 2*2 / 2
        #               = 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l2 output dim = 64 - 5 + 1 + 2*2 / 2
        #               = 32

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l3 output dim = 32 - 5 + 1 + 2*2 / 2
        #               = 16

        # push through layers 4 & 5 before pooling
        # keep channel at 64 until layer 5
        # use 3x3 convolutions as the inputs are only 16x16 now

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        # l4 output dim = 16 - 3 + 1 + 2*2 / 1
        #               = 16

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l5 output dim = 16 - 3 + 1 + 2*2 / 2
        #               = 8

        self.fc = nn.Linear(8*8*128, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# Baseline with 7 Layers
class PetsConvNetL7(nn.Module):
    def __init__(self, num_classes=4):
        super(PetsConvNetL7, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l1 output dim = 128 - 5 + 1 + 2*2 / 2
        #               = 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l2 output dim = 64 - 5 + 1 + 2*2 / 2
        #               = 32

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l3 output dim = 32 - 5 + 1 + 2*2 / 2
        #               = 16

        # push through layers 4 & 5 before pooling
        # keep channel at 64 until layer 5
        # use 3x3 convolutions as the inputs are only 16x16 now

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        # l4 output dim = 16 - 3 + 1 + 2*1 / 1
        #               = 16

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l5 output dim = 16 - 3 + 1 + 2*1 / 2
        #               = 8

        # push through layer 6 and increase channels again at layer 7
        # don't downsample again until layer 7
        # use 3x3 convolutions as the inputs are only 8x8 now
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        # l6 output dim = 8 - 3 + 1 + 2*1 / 1
        #               = 8

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # l7 output dim = 8 - 3 + 1 + 2*1 / 2
        #               = 4

        self.fc = nn.Linear(4 * 4 * 256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out