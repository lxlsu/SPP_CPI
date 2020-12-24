"""
Does not contain fingerprint information

The doc2VEC feature of the protein were directly linked to the compound characteristics.
"""

import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from spp_layer import spatial_pyramid_pool

if torch.cuda.is_available():
    device = torch.device("cuda")  # "cuda:0"
else:
    device = torch.device("cpu")


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5,
                     stride=stride, padding=2, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv5x5(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.elu(out)
        return out


class SPP_CPI(torch.nn.Module):
    """
    The class is an implementation of the SPP_CPI model including regularization and without pruning.
    Slight modifications have been done for speedup

    """

    def __init__(self, args, block, train=False):

        super(SPP_CPI, self).__init__()
        self.batch_size = args['batch_size']
        self.train_f = train
        self.r = args['r']
        self.type = args['task_type']
        self.in_channels = args['in_channels']
        self.dim_pro = args['protein_fc']
        self.output_num = [4, 2, 1]
        self.spp_out_dim = args['spp_out_dim']
        self.finger = 0
        self.hidden_nodes = args['hidden_nodes']

        # self.blocks_num = [args['block_num1'], args['block_num2']]

        self.fc1 = nn.Linear(args['cnn_channel_block2'] * (1+4+16), self.hidden_nodes)
        self.fc2 = nn.Linear(self.hidden_nodes, self.spp_out_dim)

        self.linear_first_seq = torch.nn.Linear(args['cnn_channel_block2'], args['d_a'])
        self.linear_second_seq = torch.nn.Linear(args['d_a'], self.r)

        # cnn
        self.conv = conv3x3(1, self.in_channels)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.elu = nn.ELU(inplace=False)
        self.layer1 = self.make_layer(block, args['cnn_channel_block1'], args['block_num1'])
        self.layer2 = self.make_layer(block, args['cnn_channel_block2'],  args['block_num2'])

        self.linear_final_step = torch.nn.Linear(self.spp_out_dim + self.dim_pro + self.finger, args['fc_final'])
        self.linear_final = torch.nn.Linear(args['fc_final'], args['n_classes'])

        # self.hidden_state = self.init_hidden()
        # self.seq_hidden_state = self.init_hidden()

        self.fc_protein_3 = nn.Sequential(
            nn.Linear(args['protein_input_dim'], 512),
            # nn.BatchNorm1d(256),   # 0425
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            # nn.BatchNorm1d(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(256, self.dim_pro)
        )

        self.fc_protein_2 = nn.Sequential(
            nn.Linear(args['protein_input_dim'], 512),
            # nn.BatchNorm1d(256),   # 0425
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),

            nn.Linear(512,  self.dim_pro),
            nn.Dropout(0.3),
            # nn.BatchNorm1d(128, 64),
            nn.ReLU(inplace=True),
        )

        self.fc_compound = nn.Sequential(
            nn.Linear(args['cnn_channel_block2'] * (1+4+16), self.hidden_nodes),
            # nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),

            nn.Linear(self.hidden_nodes, self.spp_out_dim),
            # nn.Dropout(0.2),
            nn.ReLU(inplace=True),
        )


    def softmax(self, input, axis=1):
        """
        Softmax applied to axis=n
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied

        Returns:
            softmaxed tensors
        """
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def make_layer(self, block, out_channels, block_num, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, block_num):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    # x1 = smiles , x3 = proteins
    def forward(self, x1, finger, x3):

        protein = x3.view(x3.size(0), -1).to(device)
        pt_feature = self.fc_protein_2(protein)
        pt_feature = pt_feature.view(pt_feature.size(0), -1)  # batch_size*vec_len

        x1 = torch.unsqueeze(x1, 1)
        pic = self.conv(x1)

        pic = self.bn(pic)
        pic = self.elu(pic)
        pic = self.layer1(pic)
        pic = self.layer2(pic)

        spp = spatial_pyramid_pool(pic, pic.size(0), [int(pic.size(2)), int(pic.size(3))], self.output_num)

        fc1 = F.relu(self.fc1(spp))
        fc2 = F.relu(self.fc2(fc1))

        sscomplex = torch.cat([fc2, pt_feature], dim=1)
        sscomplex = torch.relu(self.linear_final_step(sscomplex))

        if not bool(self.type):
            pred = self.linear_final(sscomplex)
            pic_output = torch.sigmoid(pred)
            return pic_output
