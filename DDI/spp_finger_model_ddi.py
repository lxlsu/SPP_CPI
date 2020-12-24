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


class DrugVQA(torch.nn.Module):
    """
    The class is an implementation of the DrugVQA model including regularization and without pruning.
    Slight modifications have been done for speedup

    """

    def __init__(self, args, block):
        """
        Initializes parameters suggested in paper

        args:
            batch_size  : {int} batch_size used for training
            lstm_hid_dim: {int} hidden dimension for lstm
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            n_chars_smi : {int} voc size of smiles
            n_chars_seq : {int} voc size of protein sequence
            dropout     : {float}
            in_channels : {int} channels of CNN block input
            cnn_channels: {int} channels of CNN block
            cnn_layers  : {int} num of layers of each CNN block
            emb_dim     : {int} embeddings dimension
            fc_final   : {int} hidden dim for the output dense
            task_type   : [0,1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes

        Returns:
            self
        """
        super(DrugVQA, self).__init__()
        self.batch_size = args['batch_size']

        # self.r = args['r']
        self.type = args['task_type']
        self.in_channels = args['in_channels']
        # self.dim_pro = args['protein_fc']
        self.output_num = [4, 2, 1]
        self.spp_out_dim = args['spp_out_dim']
        self.finger_len = 0

        self.fc1 = nn.Linear(args['cnn_channels'] * 21, 1024)
        self.fc2 = nn.Linear(1024, self.spp_out_dim)

        # self.linear_first_seq = torch.nn.Linear(args['cnn_channels'], args['d_a'])
        # self.linear_second_seq = torch.nn.Linear(args['d_a'], self.r)

        # cnn
        self.conv_drug1 = conv3x3(1, self.in_channels)
        self.bn_drug1 = nn.BatchNorm2d(self.in_channels)
        self.elu_drug1 = nn.ELU(inplace=False)
        self.layer1_drug1 = self.make_layer(block, args['cnn_channels'], args['cnn_layers'])
        self.layer2_drug1 = self.make_layer(block, args['cnn_channels'], args['cnn_layers'])

        self.conv_drug2 = conv3x3(1, self.in_channels)
        self.bn_drug2 = nn.BatchNorm2d(self.in_channels)
        self.elu_drug2 = nn.ELU(inplace=False)
        self.layer1_drug2 = self.make_layer(block, args['cnn_channels'], args['cnn_layers'])
        self.layer2_drug2 = self.make_layer(block, args['cnn_channels'], args['cnn_layers'])

        self.linear_final_step = torch.nn.Linear(self.spp_out_dim*2 + self.finger_len, args['fc_final'])
        self.linear_final = torch.nn.Linear(args['fc_final'], args['n_classes'])


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

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    # x1 = smiles , x2 = smiles
    def forward(self, x1, x2):

        x1 = torch.unsqueeze(x1, 1)
        drug1 = self.conv_drug1(x1)
        # print(x1.shape, pic.shape)
        drug1 = self.bn_drug1(drug1)
        drug1 = self.elu_drug1(drug1)
        drug1 = self.layer1_drug1(drug1)
        drug1 = self.layer2_drug1(drug1)

        drug2 = torch.unsqueeze(x2, 1)
        drug2 = self.conv_drug2(drug2)
        # print(x1.shape, pic.shape)
        drug2 = self.bn_drug2(drug2)
        drug2 = self.elu_drug2(drug2)
        drug2 = self.layer1_drug2(drug2)
        drug2 = self.layer2_drug2(drug2)

        # print(pic.shape, "pic.shape")

        spp_drug1 = spatial_pyramid_pool(drug1, drug1.size(0), [int(drug1.size(2)), int(drug1.size(3))],
                                         self.output_num)
        spp_drug2 = spatial_pyramid_pool(drug2, drug2.size(0), [int(drug2.size(2)), int(drug2.size(3))],
                                         self.output_num)
        # print(spp.shape, "spp.shape")
        fc1_drug1 = F.relu(self.fc1(spp_drug1))
        fc2_drug1 = F.relu(self.fc2(fc1_drug1))

        fc1_drug2 = F.relu(self.fc1(spp_drug2))
        fc2_drug2 = F.relu(self.fc2(fc1_drug2))

        # print(fc2.shape, "fc2.shape")

        sscomplex = torch.cat([fc2_drug1, fc2_drug2], dim=1)
        sscomplex = torch.relu(self.linear_final_step(sscomplex))

        if not bool(self.type):
            pred = self.linear_final(sscomplex)
            pic_output = torch.sigmoid(pred)
            return pic_output

