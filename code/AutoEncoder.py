import os
import re
import pickle
import numpy as np
from tqdm import tqdm
import torch
torch.manual_seed(42)
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import config
plt.switch_backend('agg')
from sklearn.model_selection import train_test_split

c1 = "#20639B"
c2 = "#ED553B"

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1,256,40)
                        nn.Conv2d(in_channels=1,
                               out_channels=8,
                               kernel_size=5,
                               stride=3,
                               padding=1),  # output shape (8,25,39)
                        nn.BatchNorm2d(8),
                        nn.ReLU()
                        )
        self.conv2 = nn.Sequential(  # input shape (8,24,20)
                        nn.Conv2d(in_channels=8,
                               out_channels=16,
                               kernel_size=5,
                               stride=3,
                               padding=1),  # output shape (16,11,9)
                        nn.BatchNorm2d(16),
                        nn.ReLU()
                        )

        self.fc1 = nn.Sequential(
                        nn.Linear(16*28*4, 128),
                        #nn.BatchNorm2d(1),
                        nn.ReLU()
                        )

        self.fc2 = nn.Sequential(
                        nn.Linear(128, 16),
                        #nn.BatchNorm2d(1),
                        nn.ReLU()
                        )


        self.fc3 = nn.Sequential(
                        nn.Linear(16, 128),
                        #nn.BatchNorm2d(128),
                        nn.ReLU()
                        )

        self.fc4 = nn.Sequential(
                        nn.Linear(128, 16*28*4),
                        nn.ReLU()
                        )

        self.conv3 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=16,
                                out_channels=8,
                                kernel_size=5,
                                stride=3,
                                padding=1,
                                output_padding=(1,1)),
                        nn.BatchNorm2d(8),
                        nn.ReLU()
                        )
        self.conv4 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=8,
                                out_channels=1,
                                kernel_size=5,
                                stride=3,
                                padding=1,
                                output_padding=(1,1)),
                        nn.BatchNorm2d(1),
                        nn.ReLU()
                        )

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # encode
        x = self.conv1(x)
        #print(x.size())
        x = self.conv2(x)
        #print(x.size())
        x = x.view(x.size()[0], 1, 1, -1)
        x = self.dropout(x)
        #print(x.size())
        x = self.fc1(x)
        x = self.dropout(x)
        encoded = self.fc2(x)
        #print(encoded.size())
        
        # decode
        x = self.dropout(encoded)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.dropout(x)
        #print(x.size())
        x = x.view(-1,16,28,4)
        x = self.conv3(x)
        #print(x.size())
        decoded = self.conv4(x)
        #print(decoded.size())

        return encoded, decoded

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), '../out/AE.pkl')
        self.val_loss_min = val_loss

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def save_decoded_image(spec, name):
    t = config.t; f = config.f
    fig = plt.figure(figsize=(10,10))
    plt.pcolormesh(t, f, spec)
    plt.colorbar()
    plt.title('STFT Magnitude ({})'%name.split('/','.')[-2], fontsize=30)
    plt.ylabel('Frequency [Hz]', fontsize=30)
    plt.xlabel('Time [sec]', fontsize=30)
    plt.xticks(fontsize=30); plt.yticks(fontsize=30)
    save_image(fig, name)


def get_data(num):
    """
    Get train data, Check the dimension
    """
    dataset = []
    path = "/home/husir/Desktop/unsupervised_clustering_code/events/spectrograms/data/"
    events = os.listdir(path)
    if num == 0:
        num = len(events)

    file_name = []
    for i in tqdm(range(num)):
        spec = np.loadtxt(path+events[i])
        print(spec.shape)
        if spec.shape != (256,40):
            print('Wrong Dimension!')
        else:
            file_name.append(events[i])
            dataset.append(spec)
    print('Total train samples: {}'.format(len(dataset)))

    dataset = np.array(dataset)
    dataset = dataset.reshape(-1, 1, dataset.shape[1], dataset.shape[2])
    dataset = torch.tensor(dataset).float()

    print('sample shape: {}'.format(dataset.shape))
    return dataset, file_name


if __name__ == '__main__':
    # Model parameters
    epochs = 300
    batch_size = 128
    lr = 1e-4
    patience = 20

    # Check if GPU is active
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)
    

    # Read in dataset and divide into train & validation dataset
    eventnum = 0
    data_set, file_name = get_data(eventnum)
    trainset, validset = train_test_split(data_set, test_size=0.3, shuffle=True)
    print('Train set has {} samples'.format(len(trainset)))
    print('Validation set has {} samples'.format(len(validset)))

    train_loader = DataLoader(dataset = TensorDataset(trainset),
                    batch_size = batch_size,
                    shuffle=True,
                    num_workers=10,
                    )
    val_loader = DataLoader(dataset = TensorDataset(validset),
                    batch_size = 1,
                    num_workers=10,
                    )


    # Initialize the model, optimizer and loss function
    model = AE()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(model)
    print(get_parameter_number(model))

    # Train and validation loss list
    loss_stats = {'train': [], 'val': []}

    # Train model
    print("Begin training:")
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for e in range(1, epochs+1):
        # TRAINING
        train_epoch_loss = 0
        model.train()
        for batch_idx, train_batch in enumerate(train_loader,0):
            # Here we don't have labels
            train_data = train_batch[0]
            train_data = train_data.to(device)
            #print(train_data.size())
            optimizer.zero_grad()
            _, train_recon = model(train_data)

            train_loss = criterion(train_data, train_recon)
            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()

        # VALIDATION
        with torch.no_grad():
            val_epoch_loss = 0
            model.eval()
            for batch_idx, val_batch in enumerate(val_loader,0):
                val_data = val_batch[0]
                val_data = val_data.to(device)

                _, val_recon = model(val_data)

                val_loss = criterion(val_data, val_recon)
                val_epoch_loss += val_loss.item()

        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        val_avr = val_epoch_loss/len(val_loader)
        loss_stats['val'].append(val_epoch_loss/len(val_loader))

        print(f'Epoch {e+0:02} of {epochs} | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}')
        
        # early stopping evaluation
        early_stopping(val_avr, model)
        if early_stopping.early_stop:
            print('Early stopping')
            break

    # Visualize Loss
    fig = plt.figure(figsize=(10,7))
    plt.plot(loss_stats['train'], c=c1, label='Train')
    plt.plot(loss_stats['val'], c=c2, label='Validation')
    plt.title('Train-Val Loss/Epoch', fontsize = 20)
    plt.xlabel('Epochs', fontsize = 15) 
    plt.ylabel('Loss', fontsize = 15)
    plt.legend()
    plt.savefig('../out/Loss.png')
    

    # Save the model
    torch.save(model, '../out/AE.pt')

    print('----------------------Training finish---------------------')

    # Load trained model
    model = AE()
    model = torch.load('../out/AE.pt').to(device)
    model.eval()

    # Get the latent features
    event_name = []
    for i in range(len(file_name)):
        event_name.append({'sta':file_name[i][4:7], 'starttime':re.split('_|Z-',file_name[i])[1], 'duration':re.split('_|Z-',file_name[i])[2].rstrip('s.txt')})
    event_name = np.array(event_name).reshape((-1,1))
    #print(event_name)

    data_loader = DataLoader(dataset = TensorDataset(data_set),
                    batch_size = 1,
                    )

    encoded_set = []; decoded_set = []
    loss_logging = []
    for idx, data_batch in enumerate(data_loader,0):
        test_data = data_batch[0]
        test_data = test_data.to(device)
        encoded, decoded = model(test_data)
        loss = criterion(test_data, decoded) # ...

        encoded = encoded.cpu().detach().numpy()  # move to cpu and tranfer to ndarray
        decoded = decoded.cpu().detach().numpy()
        loss = loss.cpu().detach().numpy()
        encoded_set.append(encoded)
        decoded_set.append(decoded)
        loss_logging.append(loss)


    ori = data_set.cpu().detach().numpy()
    feature = np.array(encoded_set)
    recon = np.array(decoded_set)
    loss_logging = np.array(loss_logging)
    print('input: {}'.format(ori.shape))  # ( ,1,78,40)
    print('encoded: {}'.format(feature.shape))  # ( ,1,4)
    print('reconstructed: {}'.format(recon.shape))  # ( ,1,1,78,40)

    ori = ori.reshape(-1, ori.shape[-2]*ori.shape[-1]) # ( , 78*40)
    feature = feature.reshape(-1, feature.shape[-1]) # ( , 4)
    recon = recon.reshape(-1, recon.shape[-2]*recon.shape[-1]) # ( , 78*40)
    loss_logging = loss_logging.reshape(-1, 1) # ( , 1)

    #print(event_name.shape, feature.shape, loss_logging.shape)
    catalog = np.hstack((event_name, feature, loss_logging))
    #print(catalog)
    with open('../out/features_loss.dat', 'wb') as f:
        pickle.dump(catalog, f)

    print('-------------------------Features extraction complete----------------------------')
