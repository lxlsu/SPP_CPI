import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    """
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    """
    # spatial_pyramid_pool(pic, pic.size(0), [int(pic.size(2)), int(pic.size(3))], self.output_num)

    for i in range(len(out_pool_size)):
        h, w = previous_conv_size


        kernel_size = (math.ceil(h / out_pool_size[i]), math.ceil(w / out_pool_size[i]))
        stride = (math.floor(h / out_pool_size[i]), math.floor(w / out_pool_size[i]))
        pooling = (math.floor((kernel_size[0] * out_pool_size[i] - h + 1) / 2),
                   math.floor((kernel_size[1] * out_pool_size[i] - w + 1) / 2))

        # update input data with padding
        zero_pad = torch.nn.ZeroPad2d((pooling[1], pooling[1], pooling[0], pooling[0]))
        x_new = zero_pad(previous_conv)

        # update kernel and stride
        h_new = 2 * pooling[0] + h
        w_new = 2 * pooling[1] + w

        kernel_size = (math.ceil(h_new / out_pool_size[i]), math.ceil(w_new / out_pool_size[i]))
        stride = (math.floor(h_new / out_pool_size[i]), math.floor(w_new / out_pool_size[i]))

        max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        # max_pool = nn.MaxPool2d(kernel_size=(h_wid, w_wid), stride=(h_str, w_str), padding=(h_pad, w_pad))
        x = max_pool(x_new)
        # print(h_wid, w_pad, x.shape, "x.shape")
        if i == 0:
            spp = x.view(num_sample, -1)
        else:
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp


class SPPNet(nn.Module):
    """
    A CNN model which adds spp layer so that we can input single-size tensor
    """

    def __init__(self, n_classes=102, init_weights=True):
        super(SPPNet, self).__init__()

        self.output_num = [4, 2, 1]

        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3)

        self.fc1 = nn.Linear(sum([i * i for i in self.output_num]) * 256, 4096)
        self.fc2 = nn.Linear(4096, 4096)

        self.out = nn.Linear(4096, n_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # torch.Size([N, C, H, W])
        # print(x.size())

        x = F.relu(self.conv1(x))
        x = F.local_response_norm(x, size=4)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = F.local_response_norm(x, size=4)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        spp = spatial_pyramid_pool(x, x.size(0), [int(x.size(2)), int(x.size(3))], self.output_num)

        fc1 = F.relu(self.fc1(spp))
        fc2 = F.relu(self.fc2(fc1))

        output = self.out(fc2)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
