import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
CNN
"""


class SMILES_CNN(nn.Module):
    def __init__(self, n_features, n_chars):
        super(SMILES_CNN, self).__init__()

        self.n_features = n_features
        self.n_chars = n_chars

        self.conv1 = nn.Conv2d(1, 8, (3, 3), padding="same")
        self.conv2 = nn.Conv2d(8, 8, (3, 3), padding="same")
        self.conv3 = nn.Conv2d(8, 8, (3, 3), padding="same")

        self.conv4 = nn.Conv2d(8, 16, (3, 3), padding="same")
        self.conv5 = nn.Conv2d(16, 16, (3, 3), padding="same")
        self.conv6 = nn.Conv2d(16, 16, (3, 3), padding="same")

        self.conv7 = nn.Conv2d(16, 32, (3, 3), padding="same")
        self.conv8 = nn.Conv2d(32, 32, (3, 3), padding="same")
        self.conv9 = nn.Conv2d(32, 32, (3, 3), padding="same")

        self.dense1 = nn.Linear(32 * self.n_chars * self.n_features, 1)
        self.relu1 = nn.ReLU(inplace=False)

        self.shortcut = nn.Identity()
        self.block = nn.Identity()

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=False)
        res = self.shortcut(x)
        x = self.block(x)
        x = F.relu(self.conv2(x), inplace=False)
        x = F.relu(self.conv3(x), inplace=False)
        x = x + res

        x = F.relu(self.conv4(x), inplace=False)
        res = self.shortcut(x)
        x = self.block(x)
        x = F.relu(self.conv5(x), inplace=False)
        x = F.relu(self.conv6(x), inplace=False)
        x = x + res

        x = F.relu(self.conv7(x), inplace=False)
        res = self.shortcut(x)
        x = self.block(x)
        x = F.relu(self.conv8(x), inplace=False)
        x = F.relu(self.conv9(x), inplace=False)
        x = x + res

        x = x.view(1, 32 * self.n_chars * self.n_features)
        x = self.dense1(x)

        return x


def train_CNN(model, train_dataloader):

    model = model.cuda()
    model.train()

    lr = 1e-4
    epochs = 1000

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    idx = 0

    for epoch in range(100):

        for SMILE, activity in train_dataloader:

            SMILE, activity = SMILE.float(), activity.float()
            SMILE, activity = SMILE.cuda(), activity.cuda()

            optimizer.zero_grad()

            output = model(SMILE)

            loss = criterion(output, activity)

            loss.backward()
            optimizer.step()

            # Periodic Updates
            if idx % 500 == 0:
                print(
                    "Epoch: {}\t{}/{}\tLoss:{}".format(
                        epoch, idx, len(train_dataloader), loss
                    )
                )
            idx += 1

        if epoch % 5 == 0:
            torch.save(
                model.state_dict(),
                "/content/drive/MyDrive/challenge/trained/{}/{}_epochs.pt".format(
                    kinase_name, epoch
                ),
            )


"""
Transformer
"""


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=121):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Encode position according to original Attention is All You Need paper
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim)
        )
        pe = torch.zeros(max_len, 1, model_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class SMILESTransformerEncoder(nn.Module):
    def __init__(self, n_attn_heads, n_encoder_layers):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=60, nhead=n_attn_heads, dim_feedforward=512, batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)

    def forward(self, x):
        return self.encoder(x)


class SMILESSequenceModel(nn.Module):
    def __init__(self, n_attn_heads, n_encoder_layers):
        super().__init__()
        self.position_encoder = PositionalEncoding(60)
        self.transfomer_encoder = SMILESTransformerEncoder(
            n_attn_heads, n_encoder_layers
        )
        self.dense_output = nn.Linear(121 * 60, 1)

    def forward(self, x):
        x = self.position_encoder(x)
        memory = self.transfomer_encoder(x)
        memory = memory.view(-1, 121 * 60)
        x = self.dense_output(memory)

        return x


def train_transformer(model, train_dataloader):

    model.cuda()
    model.train()

    lr_max = 1e-4
    epochs = 1000

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    warmup_done = False
    warmup_iters = 4000

    idx = 0

    for epoch in range(10000):

        for SMILE, activity in kinase_encoded_dataloader:

            SMILE, activity = SMILE.float(), activity.float()
            SMILE, activity = SMILE.cuda(), activity.cuda()

            optimizer.zero_grad()
            output = model(SMILE)[:, 0]

            loss = criterion(output, activity)

            loss.backward()
            optimizer.step()

            if warmup_done == False:
                optimizer.param_groups[0]["lr"] = (
                    optimizer.param_groups[0]["lr"] + (lr_max / warmup_iters)
                )
                if optimizer.param_groups[0]["lr"] > lr_max:
                    warmup_done = True

            if idx % 10 == 0:
                print(
                    "Epoch: {}\t{}/{}\tLoss:{}".format(
                        epoch, idx, len(kinase_encoded_dataloader), loss
                    )
                )
            idx += 1

        if epoch % 50 == 0:
            torch.save(
                model.state_dict(),
                "/content/drive/MyDrive/challenge/trained/all_kinase_transformer/{}_epochs.pt".format(
                    epoch
                ),
            )
