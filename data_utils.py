import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models import SMILES_CNN


def one_hot_smiles(smi, maxlen=120):
    """
    Credit for this function to https://franky07724-57962.medium.com/
    """
    SMILES_CHARS = [
        " ",
        "#",
        "%",
        "(",
        ")",
        "+",
        "-",
        ".",
        "/",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "=",
        "@",
        "A",
        "B",
        "C",
        "F",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "R",
        "S",
        "T",
        "V",
        "X",
        "Z",
        "[",
        "\\",
        "]",
        "a",
        "b",
        "c",
        "e",
        "g",
        "i",
        "l",
        "n",
        "o",
        "p",
        "r",
        "s",
        "t",
        "u",
    ]

    smi2index = dict((c, i) for i, c in enumerate(SMILES_CHARS))
    index2smi = dict((i, c) for i, c in enumerate(SMILES_CHARS))

    X = np.zeros((maxlen, len(SMILES_CHARS)))

    for i, c in enumerate(smi):
        X[i, smi2index[c]] = 1
    return X


# Make a function that adds the one-hot encoding for the kinases
def add_kinase_encoding(one_hot_SMILES, kinase):

    n_tokens = one_hot_SMILES.shape[0]
    blank = np.zeros((n_tokens + 1, 60))
    blank[1:, 4:] = one_hot_SMILES

    if kinase == "JAK1":
        blank[0, 0] = 1
    if kinase == "JAK2":
        blank[0, 1] = 1
    if kinase == "JAK3":
        blank[0, 2] = 1
    if kinase == "TYK2":
        blank[0, 3] = 1

    return blank


class SMILES_Dataset(Dataset):
    def __init__(self, list_of_indices, dataframe, encode_kinase=False):

        # Get indices from main dataframe for given kinase
        self.list_of_indices = list_of_indices
        self.dataframe = dataframe
        self.encode_kinase = encode_kinase

    def __len__(self):
        return len(self.list_of_indices)

    def __getitem__(self, idx):

        # Get row from dataframe
        row = self.dataframe.iloc[idx]

        # Convert SMILES to One Hot
        oh_smiles = one_hot_smiles(row["SMILES"])

        kinase_name = row["Kinase_name"]

        if self.encode_kinase == True:
            oh_smiles = add_kinase_encoding(oh_smiles, kinase_name)

        oh_smiles_pt = torch.from_numpy(oh_smiles)

        # Convert activity to proportion
        activity = (float(row["measurement_value"]) - 6.0) / 6.0

        return oh_smiles_pt, activity


def make_kinase_specific_dataset(kinase_name, data):
    indices = data.index[data["Kinase_name"] == kinase_name].tolist()
    dataset = SMILES_Dataset(indices[: int(len(indices) * 0.8)], data)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return train_dataloader


def get_CNN_test_data(dataframe, kinase):
    test_set_indices = dataframe.index[dataframe["Kinase_name"] == kinase].tolist()
    test_set_indices = test_set_indices[int(len(test_set_indices) * 0.8) :]
    pki_indices = [
        i for i in test_set_indices if dataframe.iloc[i]["measurement_type"] == "pKi"
    ]
    return pki_indices


def make_kinase_encoded_SMILES_dataloader(dataframe, batch_size=1):
    kinase_encoded_dataset = SMILES_Dataset(
        list(range(int(0.8 * len(dataframe)))), dataframe, encode_kinase=True
    )
    kinase_encoded_dataloader = DataLoader(
        kinase_encoded_dataset, batch_size=batch_size, shuffle=True
    )
    return kinase_encoded_dataloader


def evaluate_model(model, data_indices, dataframe, encode_kinase=False):

    true_values = []
    estimated_values = []

    model.eval()
    model.cuda()

    dataset = SMILES_Dataset(
        data_indices[: int(len(data_indices) * 0.8)],
        dataframe,
        encode_kinase=encode_kinase,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for SMILE, true_value in dataloader:

        SMILE, true_value = SMILE.float(), true_value.float()
        SMILE, true_value = SMILE.cuda(), true_value.cuda()

        output = model(SMILE)

        estimated_values.append(float(output.data[0]))
        true_values.append(float(true_value.data[0]))

    return np.asarray(true_values) * 6.0 + 6.0, np.asarray(estimated_values) * 6.0 + 6.0


def get_MAE(s1, s2):
    return np.mean(abs(s1 - s2))
