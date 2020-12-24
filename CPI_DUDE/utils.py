import torch
from torch.autograd import Variable
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom as molDG


# from torch.utils.data import Dataset, DataLoader


def get_3DDistanceMatrix(trainFoldPath):
    """
    obtain distance matrix
    Args:
        trainFoldPath:

    Returns:

    """
    with open(trainFoldPath, 'r') as f:
        trainCpi_list = f.read().strip().split('\n')
    trainDataSet = [cpi.strip().split()[0] for cpi in trainCpi_list]
    smilesDataset = []
    for smile in trainDataSet:
        mol = Chem.MolFromSmiles(smile)
        bm = molDG.GetMoleculeBoundsMatrix(mol)
        # print(len(bm))
        # mol2 = Chem.AddHs(mol) # 加氢
        AllChem.EmbedMolecule(mol, randomSeed=1)
        dm = AllChem.Get3DDistanceMatrix(mol)
        dm_tensor = torch.FloatTensor([sl for sl in dm])
        # print(len(dm))


def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


def readLinesStrip(lines):
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip('\n')
    return lines


def getTrainDataSet(trainFoldPath):
    with open(trainFoldPath, 'r') as f:
        trainCpi_list = f.read().strip().split('\n')
    trainDataSet = [cpi.strip().split() for cpi in trainCpi_list]
    return trainDataSet  # [[smiles, sequence, interaction],.....]


def getTestProteinList(testFoldPath):
    testProteinList = readLinesStrip(open(testFoldPath).readlines())[0].split()
    return testProteinList  # ['kpcb_2i0eA_full','fabp4_2nnqA_full',....]


if __name__ == "__main__":
    tFoldPath = ""
    get_3DDistanceMatrix(tFoldPath)
