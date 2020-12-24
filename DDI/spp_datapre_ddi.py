import sys
from utils import *
from torch.utils.data import Dataset, DataLoader
# torch.cuda.set_device(1)


# torch.cuda.set_device(1)
if torch.cuda.is_available():
    device = torch.device("cuda")   # "cuda:0"
else:
    device = torch.device("cpu")


class ProDataset_smiles(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataSet, seqContactDict):
        self.dataSet = dataSet  # list:[[smile,seq,label],....]
        self.len = len(dataSet)
        self.dict = seqContactDict  # dict:{seq:contactMap,....}
        self.properties = [int(x[2]) for x in dataSet]  # labels
        self.property_list = list(sorted(set(self.properties)))

    def __getitem__(self, index):
        smiles, seq, label = self.dataSet[index]
        contactMap = self.dict[seq]
        # return smiles, contactMap, int(label)
        return smiles, seq, int(float(label))

    def __len__(self):
        return self.len

    def get_properties(self):
        return self.property_list

    def get_property(self, id):
        return self.property_list[id]

    def get_property_id(self, property):
        return self.property_list.index(property)


class ProDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self,dataSet, druga, drugb):
        self.dataSet = dataSet    # list:[[smile,seq,label],....]
        self.len = len(dataSet)

        self.drugA = druga
        self.drugB = drugb

    def __getitem__(self, index):
        _, _, label = self.dataSet[index]

        dm_drug1 = self.drugA[index]
        dm_drug2 = self.drugB[index]

        return dm_drug1, dm_drug2, int(float(label))

    def __len__(self):
        return self.len


def load_tensor(file_name, dtype):
    if "protein" in file_name:
        return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]
        # return [dtype(d).transpose(1,0) for d in np.load(file_name + '.npy')]
    else:
        return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]
