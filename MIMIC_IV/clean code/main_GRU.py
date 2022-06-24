import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import GRU
import pandas as pd 
import torch.nn as nn
from preprocessing import Preprocessing
from torch.utils.data import TensorDataset, DataLoader
import warnings
import torch.utils.data as utils
import time
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
warnings.filterwarnings("ignore")

time.clock = time.time

#Parameters
nb_hours = 24
random_state = 1
TBI_split = False
tuning = False
SHAP = False
imputation = 'carry_forward'
model_name = 'Stacking'
lr = 0.0001
learning_rate_decay = 7 
n_epochs = 14
batch_size = 16

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('Device', device)  
def train(train_loader, dev_loader, test_loader, learn_rate, hidden_dim=512, EPOCHS=5, model_type="GRU"):
    
    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]
    hidden_dim = 256
    output_dim = 1
    n_layers = 49
    # Instantiating the models
    if model_type == "GRU":
        model = GRU.GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    else:
        model = GRU.LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)
    
    # Defining loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    # Start training loop
    for epoch in range(1,EPOCHS+1):
        start_time = time.clock()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        losses, acc = [], []
        label, pred = [], []
        epoch_losses = []
        y_pred_col = []
        
        model.train()
        for x, train_label in train_loader:
            counter += 1
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()
            
            y_pred, h = model(x.to(device).float(), h)
            y_pred = torch.squeeze(y_pred)
            y_pred_col.append(y_pred)

            pred.append(y_pred > 0.5)
            label.append(train_label)
            loss = criterion(y_pred.float(), train_label.float())
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    train_label)
            )
            losses.append(loss.item())

            #backward pass 
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
        
        train_acc = torch.mean(torch.cat(acc).float())
        train_loss = np.mean(losses)
        train_pred_out = pred
        train_label_out = label
        model.eval()
        
        #validation set 
        losses, acc = [], []
        label, pred = [], []
        for dev_data, dev_label in dev_loader:
            # Forward pass : Compute predicted y by passing train data to the model
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            y_pred, h = model(dev_data.to(device).float(), h)
            y_pred = torch.squeeze(y_pred)
            
            # Save predict and label
            pred.append(y_pred)
            label.append(dev_label)

            # Compute loss
            loss = criterion(y_pred.float(), dev_label.float())
            acc.append(torch.eq((torch.sigmoid(y_pred).data > 0.5).float(),dev_label))
            losses.append(loss.item())
            
        dev_acc = torch.mean(torch.cat(acc).float())
        dev_loss = np.mean(losses)
        dev_pred_out = pred
        dev_label_out = label
        current_time = time.clock()

        losses, acc = [], []
        label, pred = [], []
        model.eval()

        for test_data, test_label in test_loader:
            test_data = torch.squeeze(test_data)
            test_label = torch.squeeze(test_label)
            
            # Forward pass : Compute predicted y by passing train data to the model
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            y_pred, h = model(test_data.to(device).float(), h)
            y_pred = torch.squeeze(y_pred)
            
            # Save predict and label
            pred.append(y_pred.detach().numpy())
            label.append(test_label.detach().numpy())

            # Compute loss
            loss = criterion(y_pred.float(), test_label.float())
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    test_label)
            )
            losses.append(loss.item())
            
        test_acc = torch.mean(torch.cat(acc).float())
        test_loss = np.mean(losses)
        test_pred_out = pred
        test_label_out = label
                
        epoch_losses.append([
             train_loss, dev_loss, test_loss,
             train_acc, dev_acc, test_acc,
             train_pred_out, dev_pred_out, test_pred_out,
             train_label_out, dev_label_out, test_label_out,
         ])
        
        pred = np.asarray(pred)
        label = np.asarray(label)
        
        auc_score = roc_auc_score(label, pred)
        print("Epoch: {} Train loss: {:.4f}, Dev loss: {:.4f}, Test loss: {:.4f}, Test AUC: {:.4f}".format(
            epoch, train_loss, dev_loss, test_loss, auc_score))

    return model

def evaluate(model, test_x, test_y, label_scalers):
    model.eval()
    outputs = []
    targets = []
    start_time = time.clock()
    for i in test_x.keys():
        inp = torch.from_numpy(np.array(test_x[i]))
        labs = torch.from_numpy(np.array(test_y[i]))
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
        outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
    print("Evaluation Time: {}".format(str(time.clock()-start_time)))
    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i]-targets[i])/(targets[i]+outputs[i])/2)/len(outputs)
    print("sMAPE: {}%".format(sMAPE*100))
    return outputs, targets, sMAPE  


##data loading 
df_hourly = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_hour.csv', delimiter=',')
df_24h = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_24hour.csv', delimiter=',')
df_48h = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_48hour.csv', delimiter=',')
df_med = pd.read_csv(r"C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_med.csv", delimiter=',')
df_demographic = pd.read_csv(r"C:\Users\USER\OneDrive\Summer_project\Azure\data\demographics_mimic4.csv", delimiter=',')
features = pd.read_csv(r'MIMIC_IV\resources\features.csv', header = None)
features = features.loc[:415,1] 

pr = Preprocessing(df_hourly, df_24h, df_48h, df_med, df_demographic, nb_hours, TBI_split, random_state, imputation)
data, labels = pr.time_series_pr()
final_data = np.array(data)
final_data = np.transpose(final_data, (0,2,1))



X_train, X_test, y_train, y_test = train_test_split(final_data, labels, test_size=0.2, shuffle = True, random_state=random_state)
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

dev_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size, drop_last=True)

test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

gru_model = train(train_loader, dev_loader, test_loader, lr, model_type="GRU", EPOCHS = n_epochs)

