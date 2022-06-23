import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import GRU
import pandas as pd 
from preprocessing import Preprocessing
import warnings

warnings.filterwarnings("ignore")

#Parameters
nb_hours = 24
random_state = 1
TBI_split = False
tuning = False
SHAP = False
imputation = 'carry_forward'
model_name = 'Stacking'
hidden_size = 415
output_size = 1
input_size = 415 
num_layers = 49 # num of step or layers base on the paper

#functions
def fit(model, criterion, learning_rate,\
        train_dataloader, dev_dataloader, test_dataloader,\
        learning_rate_decay=0, n_epochs=30):
    epoch_losses = []
    
    # to check the update 
    old_state_dict = {}
    for key in model.state_dict():
        old_state_dict[key] = model.state_dict()[key].clone()
    
    for epoch in range(n_epochs):
        
        if learning_rate_decay != 0:

            # every [decay_step] epoch reduce the learning rate by half
            if  epoch % learning_rate_decay == 0:
                learning_rate = learning_rate/2
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                print('at epoch {} learning_rate is updated to {}'.format(epoch, learning_rate))
        
        # train the model
        losses, acc = [], []
        label, pred = [], []
        y_pred_col= []
        model.train()
        for train_data, train_label in train_dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
            train_data = torch.squeeze(train_data)
            train_label = torch.squeeze(train_label)
            
            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(train_data)
            
            # y_pred = y_pred[:, None]
            # train_label = train_label[:, None]
            
            #print(y_pred.shape)
            #print(train_label.shape)
            
            # Save predict and label
            y_pred_col.append(y_pred.item())
            pred.append(y_pred.item() > 0.5)
            label.append(train_label.item())
            
            #print('y_pred: {}\t label: {}'.format(y_pred, train_label))

            # Compute loss
            loss = criterion(y_pred, train_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    train_label)
            )
            losses.append(loss.item())

            # perform a backward pass, and update the weights.
            loss.backward()
            optimizer.step()

        
        train_acc = torch.mean(torch.cat(acc).float())
        train_loss = np.mean(losses)
        
        train_pred_out = pred
        train_label_out = label
        
        # save new params
        new_state_dict= {}
        for key in model.state_dict():
            new_state_dict[key] = model.state_dict()[key].clone()
            
        # compare params
        for key in old_state_dict:
            if (old_state_dict[key] == new_state_dict[key]).all():
                print('Not updated in {}'.format(key))
   
        
        # dev loss
        losses, acc = [], []
        label, pred = [], []
        model.eval()
        for dev_data, dev_label in dev_dataloader:
            # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
            dev_data = torch.squeeze(dev_data)
            dev_label = torch.squeeze(dev_label)
            
            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(dev_data)
            
            # Save predict and label
            pred.append(y_pred.item())
            label.append(dev_label.item())

            # Compute loss
            loss = criterion(y_pred, dev_label)
            acc.append(
                torch.eq(
                    (torch.sigmoid(y_pred).data > 0.5).float(),
                    dev_label)
            )
            losses.append(loss.item())
            
        dev_acc = torch.mean(torch.cat(acc).float())
        dev_loss = np.mean(losses)
        
        dev_pred_out = pred
        dev_label_out = label
        
        # test loss
        losses, acc = [], []
        label, pred = [], []
        model.eval()
        for test_data, test_label in test_dataloader:
            # Squeeze the data [1, 33, 49], [1,5] to [33, 49], [5]
            test_data = torch.squeeze(test_data)
            test_label = torch.squeeze(test_label)
            
            # Forward pass : Compute predicted y by passing train data to the model
            y_pred = model(test_data)
            
            # Save predict and label
            pred.append(y_pred.item())
            label.append(test_label.item())

            # Compute loss
            loss = criterion(y_pred, test_label)
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
        
        # print("Epoch: {} Train: {:.4f}/{:.2f}%, Dev: {:.4f}/{:.2f}%, Test: {:.4f}/{:.2f}% AUC: {:.4f}".format(
        #     epoch, train_loss, train_acc*100, dev_loss, dev_acc*100, test_loss, test_acc*100, auc_score))
        print("Epoch: {} Train loss: {:.4f}, Dev loss: {:.4f}, Test loss: {:.4f}, Test AUC: {:.4f}".format(
            epoch, train_loss, dev_loss, test_loss, auc_score))
        
        # save the parameters
        train_log = []
        train_log.append(model.state_dict())
        torch.save(model.state_dict(), './save/grud_mean_grud_para.pt')
        
        #print(train_log)
    
    return epoch_losses        


##data loading 
df_hourly = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_hour.csv', delimiter=',')
df_24h = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_24hour.csv', delimiter=',')
df_48h = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_48hour.csv', delimiter=',')
df_med = pd.read_csv(r"C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_med.csv", delimiter=',')
df_demographic = pd.read_csv(r"C:\Users\USER\OneDrive\Summer_project\Azure\data\demographics_mimic4.csv", delimiter=',')
features = pd.read_csv(r'MIMIC_IV\resources\features.csv', header = None)
features = features.loc[:415,1] 

pr = Preprocessing(df_hourly, df_24h, df_48h, df_med, df_demographic, nb_hours, TBI_split, random_state, imputation)
final_data, labels = pr.preprocess_data()
print(final_data.mean())
model = GRU.GRUD(input_size = input_size, hidden_size= hidden_size, output_size=output_size, dropout=0, dropout_type='mloss', x_mean=x_mean, num_layers=num_layers)


