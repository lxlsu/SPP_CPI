from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import sys
import pickle

sys.path.append("../code/")
from utils import *

# torch.cuda.set_device(1)
if torch.cuda.is_available():
    device = torch.device("cuda")  # "cuda:0"
else:
    device = torch.device("cpu")


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    # print(data.shape, sigma)
    # [[0.         2.75400045]
    #  [2.75400045 0.        ]]
    # if data.shape[0] == 2:

    #     print(data)
    return (data - mu) / sigma


class ProDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataSet, matrix, proteins, bits, method='n', pad=False):
        self.padding = pad
        self.dataSet = dataSet  # list:[[smile,seq,label],....]
        self.len = len(dataSet)
        # self.dict = seqContactDict  # dict:{seq:contactMap,....}
        self.properties = [int(x[2]) for x in dataSet]  # labels
        self.property_list = list(sorted(set(self.properties)))
        self.proteins = proteins
        self.matrix = matrix

        self.method = method

        self.bits = bits

    def __getitem__(self, index):
        smiles, seq, label = self.dataSet[index]
        # contactMap = self.dict[seq]     # 为tensor
        protein = self.proteins[index]
        dm = self.matrix[index]
        mol = Chem.MolFromSmiles(smiles)
        """
        # 3d距离
        mol = Chem.MolFromSmiles(smiles)

        # bm = molDG.GetMoleculeBoundsMatrix(mol)
        mol = Chem.AddHs(mol)  # 加氢
        AllChem.EmbedMolecule(mol, randomSeed=1)  # 通过距离几何算法计算3D坐标
        dm = AllChem.Get3DDistanceMatrix(mol)
        atom_nums = mol.GetNumAtoms()  # 原子数
        if self.padding and atom_nums <= 3:
            dm = np.pad(dm, 2, 'constant')
        """
        dm = dm.numpy()

        # 数据标准化, 添加于19：22
        if self.method == 'n':
            dm = normalization(dm)
        elif self.method == 's':
            dm = standardization(dm)
        else:
            dm = dm

        # 指纹信息
        #  m = Chem.MolFromSmiles(s)
        finger_info = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.bits))  # 分子指纹

        return dm, finger_info, protein, int(label)

    def __len__(self):
        return self.len

    def get_properties(self):
        return self.property_list

    def get_property(self, id):
        return self.property_list[id]

    def get_property_id(self, property):
        return self.property_list.index(property)


def load_tensor(file_name, dtype):
    if "protein" in file_name:
        return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]
        # return [dtype(d).transpose(1,0) for d in np.load(file_name + '.npy')]
    else:
        return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


class ProDataset_cmp_2018(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataSet, compound, proteins, bits, method='n', pad=False):
        self.padding = pad
        self.dataSet = dataSet  # list:[[smile,seq,label],....]
        self.len = len(dataSet)
        # self.dict = seqContactDict  # dict:{seq:contactMap,....}
        self.properties = [int(x[2]) for x in dataSet]  # labels
        self.property_list = list(sorted(set(self.properties)))
        self.proteins = proteins
        self.compound = compound

        self.method = method

        self.bits = bits

    def __getitem__(self, index):
        smiles, seq, label = self.dataSet[index]
        # contactMap = self.dict[seq]     # 为tensor
        protein = self.proteins[index]
        dm = self.compound[index]
        mol = Chem.MolFromSmiles(smiles)
        """
        # 3d距离
        mol = Chem.MolFromSmiles(smiles)

        # bm = molDG.GetMoleculeBoundsMatrix(mol)
        mol = Chem.AddHs(mol)  # 加氢
        AllChem.EmbedMolecule(mol, randomSeed=1)  # 通过距离几何算法计算3D坐标
        dm = AllChem.Get3DDistanceMatrix(mol)
        atom_nums = mol.GetNumAtoms()  # 原子数
        if self.padding and atom_nums <= 3:
            dm = np.pad(dm, 2, 'constant')
        """
        dm = dm.numpy()

        # 数据标准化, 添加于19：22
        if self.method == 'n':
            dm = normalization(dm)
        elif self.method == 's':
            dm = standardization(dm)
        else:
            dm = dm

        # 指纹信息
        #  m = Chem.MolFromSmiles(s)
        finger_info = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.bits))  # 分子指纹

        return dm, finger_info, protein, int(label)

    def __len__(self):
        return self.len

    def get_properties(self):
        return self.property_list

    def get_property(self, id):
        return self.property_list[id]

    def get_property_id(self, property):
        return self.property_list.index(property)


if __name__ == "__main__":
    arr = np.array([[0., 2.75400045], [2.75400045, 0.]])
    std = np.std(arr, axis=0)
    print(std)
