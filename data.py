# %%
import torch 
import torch.nn as nn
import numpy as np
import seaborn as sns
import pandas as pd 
import lightning as L

# %% [markdown]
# # UCI HAR

# %%
DATADIR = 'data/UCI HAR/UCI HAR Dataset/UCI HAR Dataset'

# Raw data signals
# Signals are from Accelerometer and Gyroscope
# The signals are in x,y,z directions
# Sensor signals are filtered to have only body acceleration
# excluding the acceleration due to gravity
# Triaxial acceleration from the accelerometer is total acceleration
SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z"
]

# Utility function to read the data from csv file
def _read_csv(filename):
    return pd.read_csv(filename, delim_whitespace=True, header=None)

# Utility function to load the load
def load_signals(subset):
    signals_data = []

    for signal in SIGNALS:
        filename = f'{DATADIR}/{subset}/Inertial Signals/{signal}_{subset}.txt'
        signals_data.append(
            _read_csv(filename).to_numpy()
        ) 

    # Transpose is used to change the dimensionality of the output,
    # aggregating the signals by combination of sample/timestep.
    # Resultant shape is (7352 train/2947 test samples, 128 timesteps, 9 signals)
    return np.transpose(signals_data, (1, 2, 0))

def load_y(subset):
    """
    The objective that we are trying to predict is a integer, from 1 to 6,
    that represents a human activity. We return a binary representation of 
    every sample objective as a 6 bits vector using One Hot Encoding
    (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)
    """
    filename = f'{DATADIR}/{subset}/y_{subset}.txt'
    y = _read_csv(filename)[0]

    return pd.get_dummies(y).to_numpy()

def load_data():
    """
    Obtain the dataset from multiple files.
    Returns: X_train, X_test, y_train, y_test
    """
    X_train, X_test = load_signals('train'), load_signals('test')
    y_train, y_test = load_y('train'), load_y('test')

    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()


    return X_train.swapaxes(1,2), X_test.swapaxes(1,2), y_train, y_test

# %%
x_train, x_test, y_train, y_test = load_data()

# %%
x_train.size()

# %%
num_classes = y_train.argmax(-1).unique().size()[0]
num_classes

# %%
class SConvLSTM(nn.Module):
    def __init__(self, num_classes=6):
        super(SConvLSTM, self).__init__()

        self.num_classes = num_classes 
        kernel_size = 3
        self.conv1size = 32
        self.conv2size = 64
        self.conv3size = 64

        self.conv1 = nn.Conv1d(9, self.conv1size, kernel_size=kernel_size)
        self.batch1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=kernel_size)

        self.conv2 = nn.Conv1d(32, self.conv2size, kernel_size=kernel_size)
        self.batch2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=kernel_size)

        self.conv3 = nn.Conv1d(64, self.conv3size, kernel_size=kernel_size)
        self.batch3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=kernel_size)

        self.flatten1 = nn.Flatten()

        self.lstm1 = nn.LSTM(192, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)

        self.fc1 = nn.Linear(128, num_classes)
        self.sm = nn.Softmax()


    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten1(x)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = self.fc1(x).squeeze()
        x = self.sm(x)
        return x
    
class SConvLSTMLightning(L.LightningModule):
    def __init__(self, lr=0.0001):
        super(SConvLSTMLightning, self).__init__()

        self.model = SConvLSTM()
        self.loss = nn.CrossEntropyLoss()

        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(-1) == y.argmax(-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(-1) == y.argmax(-1)).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        #acc = (y_hat.argmax() == y).float().mean()
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

class UCIHARDataModule(L.LightningDataModule):
    def __init__(self):
        super(UCIHARDataModule, self).__init__()

    
    def prepare_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = load_data()

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.x_train, self.y_train),
            batch_size=64,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.x_test, self.y_test),
            batch_size=64,
            shuffle=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.x_test, self.y_test),
            batch_size=64,
            shuffle=True
        )

# %%
for lr in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
    trainer = L.Trainer(max_epochs=20)
    model = SConvLSTMLightning(lr=lr)
    data = UCIHARDataModule()

    trainer.fit(model, data)

    accs = [] 

    for x, y in data.test_dataloader():
        y_hat = model(x)

        acc = (y_hat.argmax(-1) == y.argmax(-1)).float()
        accs.append(acc)


    accs = torch.concatenate(accs)
    trainer.logger.experiment.add_scalar('test_acc', accs.mean())
    trainer.logger.log_hyperparams({'lr': lr})


