import torch
from torch import nn
from torchvision.models import MobileNetV2
from torchvision.ops.misc import Conv2dNormActivation
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR

import pandas as pd

import os

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)



class focusDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        # remove all the files that are not power20:
        #self.img_labels = self.img_labels[self.img_labels['file_name'].str.contains('power20')]
        self.img_labels = self.img_labels.reset_index(drop=True)

        self.img_dir = img_dir
        self.transform = transform
        self.sign = lambda x: int(x >= 0)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        #label = (np.abs(np.float32(self.img_labels.iloc[idx, 1]/40)), self.sign(self.img_labels.iloc[idx, 1]))
        label = (abs(np.float32(self.img_labels.iloc[idx, 1]/40)), self.sign(self.img_labels.iloc[idx, 1]))
        if self.transform:
            image = self.transform(image)
        return image, label
    
tr = transforms.Compose([

    #transforms.Resize((224, 224), antialias=True),
    transforms.ToImage(),
    #transforms.Lambda(lambda x: x / torch.sqrt(torch.sum(x**2))),
    #transforms.ToDtype(torch.float32, scale=True),
    
    #transforms.Normalize(mean=[0.2], std=[1.0])
    # divide each image by sum of squares of pixels
])


all_data = focusDataset('dat_grid_test/file_names_and_distances.csv', 'dat_grid_test', transform=tr)
print(len(all_data))
# find and display 16 images with label 0:
indices = [i for i, x in enumerate(all_data.img_labels.iloc[:,1]) if x == 0]
indices = np.random.choice(indices, 25, replace=False)

fig, axs = plt.subplots(5, 5)
for i, idx in enumerate(indices):
    img = all_data[idx][0][0,:,:]
    fname = all_data.img_labels.iloc[idx,0]
    print(fname)
    axs[i//5, i%5].imshow(img, cmap='gray')
plt.show()



model = MobileNetV2()
model.features[0][0] = Conv2dNormActivation(1, 32, kernel_size=1, norm_layer=None, activation_layer=nn.ReLU)
model.classifier = nn.Sequential(

    nn.Dropout(0.4),
    nn.Linear(1280, 3),
    #nn.ReLU(inplace=True),
    #nn.Linear(100, 3),
)



# find all the indices of file names containing 'f3':
#all_indices = [i for i in range(len(all_data))]
#indices_test = np.random.choice(all_indices, int(len(all_indices) / 10), replace=False)
indices_test = [i for i, s in enumerate(all_data.img_labels.iloc[:,0]) if 'f3_plane2' in s]

# these indices will be used for testing
test_data = torch.utils.data.Subset(all_data, indices_test)
# the rest will be used for training
indices_train = [i for i in range(len(all_data)) if i not in indices_test]
train_data = torch.utils.data.Subset(all_data, indices_train)

test_batch_size = int(16)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)


# train
loss_fn_reg = nn.MSELoss()
loss_fn_class = nn.CrossEntropyLoss()

lr = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)#1e-6)
scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.5, verbose=True)
def train(dataloader, model, loss_fn_reg, loss_fn_class, optimizer):

    size = len(dataloader.dataset)
    train_reg_loss, train_class_loss, correct = 0, 0, 0
    num_batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y[0] = y[0].to(device)
        y[1] = y[1].to(device)

        # Compute prediction and loss
        pred = model(X)
        loss_reg = loss_fn_reg(pred[:,0], y[0])
        loss_class = loss_fn_class(pred[:,1:], y[1])
        loss = loss_reg + loss_class

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_reg_loss += loss_reg.item()
        train_class_loss += loss_class.item()

        
    train_reg_loss /= num_batches
    train_class_loss /= num_batches
    print(f'Average train regression loss: {train_reg_loss:>8f}, class loss: {train_class_loss:>8f}')
    return train_reg_loss, train_class_loss

def test(dataloader, model, loss_fn_reg, loss_fn_class):
    num_batches = len(dataloader)
    model.eval()
    reg_loss, class_loss = 0, 0
    with torch.no_grad():
        # collect all labels and predictions
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y[0] = y[0].to(device)
            y[1] = y[1].to(device)
            pred = model(X)
            reg_loss += loss_fn_reg(pred[:,0], y[0]).item()
            class_loss += loss_fn_class(pred[:,1:], y[1]).item()
            all_dist_labels = y[0].cpu().numpy()
            all_sign_labels = y[1].cpu().numpy()
            all_dist_predictions = pred[:,0].cpu().numpy()
            all_sign_predictions = torch.argmax(pred[:,1:], axis=1).cpu().numpy()
            if batch == 0:
                dist_lbl = all_dist_labels
                sign_lbl = all_sign_labels
                dist_predict = all_dist_predictions
                sign_predict = all_sign_predictions
            else:
                dist_lbl = np.concatenate((dist_lbl, all_dist_labels))
                sign_lbl = np.concatenate((sign_lbl, all_sign_labels))
                dist_predict = np.concatenate((dist_predict, all_dist_predictions))
                sign_predict = np.concatenate((sign_predict, all_sign_predictions))
    reg_loss /= num_batches
    class_loss /= num_batches
    acc = np.sum(sign_lbl == sign_predict) / len(sign_lbl)
    print(f'Average test regression loss: {reg_loss:>8f}, class loss: {class_loss:>8f}, accuracy: {acc:>8f}')
    return reg_loss, class_loss, dist_lbl, dist_predict, sign_lbl, sign_predict

epochs = 1000
fig, axs = plt.subplots(3)
plt.ion()


train_reg_losses = []
train_class_losses = []
test_reg_losses = []
test_class_losses = []

# load checkpoint if it exists:
# find potential checkpoint files:
checkpoint_files = [f for f in os.listdir('.') if f.startswith('model_checkpoint_epoch_') and f.endswith('.pth')]
last_epoch = 0
if checkpoint_files:
    # sort by epoch number and take the last one:
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = checkpoint_files[-1]
    print(f'Loading checkpoint from {checkpoint_path}')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # also update the scheduler to start from the last epoch:
    last_epoch = int(checkpoint_path.split('_')[-1].split('.')[0])
    scheduler.last_epoch = last_epoch


for t in range(last_epoch, epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    model.train()
    reg_loss, class_loss = train(train_dataloader, model, loss_fn_reg, loss_fn_class, optimizer)
    train_reg_losses.append(reg_loss)
    train_class_losses.append(class_loss)
    # every 10 epochs, save a checkpoint:
    if t % 10 == 0:
        torch.save(model.state_dict(), f'model_checkpoint_epoch_{t}.pth')
    reg_loss, class_loss, dist_lbl, dist_predict, sign_lbl, sign_predict = test(test_dataloader, model, loss_fn_reg, loss_fn_class)
    scheduler.step()
    test_reg_losses.append(reg_loss)
    test_class_losses.append(class_loss)
    
    full_labels = dist_lbl*40
    full_labels[sign_lbl == 0] = -full_labels[sign_lbl == 0]
    full_predictions = dist_predict*40
    full_predictions[sign_predict == 0] = -full_predictions[sign_predict == 0]

    axs[0].cla()
    # scatter plot of labels vs predictions with small marker size:
    negative_prediction_indices = np.where(sign_predict == 0)[0]
    positive_prediction_indices = np.where(sign_predict == 1)[0]
    axs[0].scatter(full_labels[negative_prediction_indices], full_predictions[negative_prediction_indices], s=0.3, c='r')
    axs[0].scatter(full_labels[positive_prediction_indices], full_predictions[positive_prediction_indices], s=0.3, c='b')
    min_label = np.min(full_labels)
    max_label = np.max(full_labels)
    axs[0].plot([min_label, max_label], [min_label, max_label], 'k--')

    plt.draw()
    plt.pause(0.001)
    axs[1].cla()
    axs[1].plot(train_reg_losses)
    axs[1].plot(test_reg_losses)
    plt.draw()
    plt.pause(0.001)
    axs[2].cla()
    axs[2].plot(train_class_losses)
    axs[2].plot(test_class_losses)
    plt.draw()
    plt.pause(0.001)


print('Done!')

