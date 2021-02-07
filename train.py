import torch.nn as nn  
import torch 
import numpy as np 
from train_tools import *
from data_tools import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
noisy_voice_path = '../Train/spectrogram/noisy_voice_amp_db.npy'
voice_path = '../Train/spectrogram/voice_amp_db.npy'
X = np.load(noisy_voice_path)
Y = np.load(voice_path)
Y = X-Y 

X = scaled_in(X)
Y = scaled_ou(Y)

X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
Y = Y.reshape(Y.shape[0], 1, Y.shape[1], Y.shape[2])

num = len(X)
X_train, Y_train = X[:num*9//10, :, :, :], Y[:num*9//10, :, :, :]
X_val, Y_val = X[num*9//10:, :, :, :], Y[num*9//10:, :, :, :]
X_train , Y_train = X_train.astype(np.float32), Y_train.astype(np.float32)
X_val, Y_val = X_val.astype(np.float32), Y_val.astype(np.float32)
print(X_train.shape, X_val.shape)

spec_dataset = Spec_Dataset(X_train, Y_train)
spec_dataloader = DataLoader(spec_dataset, batch_size=256, shuffle=True, pin_memory=True)
spec_valset = Spec_Dataset(X_val, Y_val)
spec_valloader = DataLoader(spec_valset, batch_size=256, shuffle=False, pin_memory=True)

same_seeds(0)
model = AE2().to(device)

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
n_epoch = 50
best_loss = 1000
len_data = len(spec_dataloader)
len_val = len(spec_valloader)

Train_loss = []
Val_loss = []

for epoch in range(n_epoch):
    model.train()
    train_loss = 0
    print('epoch {}: Training start'.format(epoch))
    for i, (dataX, dataY) in enumerate(spec_dataloader):
        print('epoch {}: {}/{}'.format(epoch, i+1, len(spec_dataloader)))
        dataX = dataX.to(device)
        dataY = dataY.to(device)
        output1, output = model(dataX)
        loss = criterion(output, dataY)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    model.eval()
    with torch.no_grad():
        val_loss = 0
        print('epoch {}: Validation start'.format(epoch))
        for i, (valX, valY) in enumerate(spec_valloader):
            print('epoch {}: {}/{}'.format(epoch, i+1, len(spec_valloader)))
            valX = valX.type(torch.cuda.FloatTensor).to(device)
            valY = valY.type(torch.cuda.FloatTensor).to(device)
            output1, output = model(valX)
            loss = criterion(output, valY)
            val_loss += loss.item()
        print('epoch [{}/{}], Train loss:{:.5f}'.format(epoch+1, n_epoch, train_loss/len_data))
        print('epoch [{}/{}], Validation loss:{:.5f}'.format(epoch+1, n_epoch, val_loss/len_val))
        Train_loss.append(train_loss/len_data)
        Val_loss.append(val_loss/len_val)
        if val_loss < best_loss: 
            best_loss = val_loss 
            torch.save(model.state_dict(), '../model/best_model5.pth')

x = [i+1 for i in range(n_epoch)]
plt.plot(x,Train_loss,label='training loss')
plt.plot(x,Val_loss,label='validation loss')
plt.legend()
plt.savefig('../picture/model5.png')
plt.show()
