import numpy as np
import torch
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
import GRU
import pandas as pd 
import torch.nn as nn
from preprocessing import Preprocessing
from torch.utils.data import TensorDataset, DataLoader
import warnings
import torch.utils.data as utils
import time
# import pickle5 as pickle
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
warnings.filterwarnings("ignore")
# import pickle5 as pickle
time.clock = time.time
import sys

#Parameters
nb_hours = 24
random_state = 42
TBI_split = False
tuning = False
SHAP = False
lr = 0.00001
lr_finetuning = 0.0001
learning_rate_decay = 7 
n_epochs = 150
n_epochs_finetuning = 800
batch_size = 10
lb = 'ABPd'
is_cuda = torch.cuda.is_available()
task = 'pressure_experiment' #Select the experiment to run 
W = 6
imputation = 'carry_forward'



##data loading 
df_24h = pd.read_csv('preprocessed_mimic4_24hour.csv', delimiter=',').sort_values(by=['stay_id'])
df_48h = pd.read_csv('preprocessed_mimic4_48hour.csv', delimiter=',').sort_values(by=['stay_id'])
df_med = pd.read_csv('preprocessed_mimic4_med.csv', delimiter=',').sort_values(by=['stay_id'])
df_demographic_augmented = pd.read_csv('demographics_mimic4_augmented.csv', delimiter=',').sort_values(by=['stay_id'])
df_demographic = pd.read_csv('demographics_mimic4.csv', delimiter=',').sort_values(by=['stay_id'])

if task in ['std', 'std_augmented', 'pressure_experiment']:
    df_hourly = pd.read_csv('preprocessed_mimic4_hour_std.csv', delimiter=',').sort_values(by=['stay_id'])
    df_hourly_augmented = pd.read_csv('preprocessed_mimic4_hour_augmented_std.csv', delimiter=',').sort_values(by=['stay_id'])
else: 
    df_hourly = pd.read_csv('preprocessed_mimic4_hour.csv', delimiter=',').sort_values(by=['stay_id'])
    df_hourly_augmented = pd.read_csv('preprocessed_mimic4_hour_augmented.csv', delimiter=',').sort_values(by=['stay_id'])


pr = Preprocessing(df_hourly_augmented, df_24h, df_48h, df_med, df_demographic_augmented, nb_hours, TBI_split, random_state, imputation)
pr_TBI = Preprocessing(df_hourly, df_24h, df_48h, df_med, df_demographic, nb_hours, TBI_split, random_state, imputation)

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('Device', device)  

#Training of our custom GRU 
def train(train_loader, dev_loader, test_loader, learn_rate, task, save = True, hidden=128, layers= 5, EPOCHS=5, model_type="GRU", severe = '', output_dim = 1, beta = 1):
    input_dim = next(iter(train_loader))[0].shape[2]
    hidden_dim = hidden
    n_layers = layers

    # Instantiating the models
    if model_type == "GRU":
        model = GRU.GRUNet(input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2, task = task)
    else:
        model = model_type
    model.to(device)
    
    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    model.train()
    print("Starting Training of {} model".format(model_type))
    ep_train_loss =[]
    ep_dev_loss = []
    # Start training loop
    for epoch in range(1,EPOCHS+1):
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        losses, mae = [], []
        label, pred = [], []
        y_pred_col = []
        
        model.train()
        for x, train_label in train_loader:
            counter += 1
            h = h.data
            model.zero_grad()
            ##Gaussian noise
            # x = x + (0.1**0.5)*torch.randn(x.shape)
            y_pred, h = model(x.to(device).float(), h)
            y_pred = torch.squeeze(y_pred)
            y_pred_col.append(y_pred)

            pred.append(y_pred.cpu().detach().numpy())
            label.append(train_label)
            loss_1 = criterion(y_pred.float()[:,0], train_label.to(device).float()[:,0])
            loss_2 = criterion(y_pred.float()[:,1], train_label.to(device).float()[:,1])
            loss = loss_1 + beta*loss_2
            m = nn.L1Loss()
            losses.append(loss.item())

            #backward pass 
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
        
        train_loss = np.mean(losses)
        model.eval()
        
        #validation set 
        losses, mae = [], []
        label, pred = [], []
        for dev_data, dev_label in dev_loader:
            # Forward pass : Compute predicted y by passing train data to the model
            h = h.data
            y_pred, h = model(dev_data.to(device).float(), h)
            y_pred = torch.squeeze(y_pred)
            # Save predict and label
            pred.append(y_pred)
            label.append(dev_label)

            # Compute loss
            loss_1 = criterion(y_pred.float()[:,0], dev_label.to(device).float()[:,0])
            loss_2 = criterion(y_pred.float()[:,1], dev_label.to(device).float()[:,1])
            loss = loss_1 + beta*loss_2
            losses.append(loss.item())
            
        dev_loss = np.mean(losses)
        losses, mae_val, mae_std = [], [], []
        rmse_val, rmse_std = [], []
        label, pred = [], []
        model.eval()

        for test_data, test_label in test_loader:
            
            test_data = torch.squeeze(test_data)
            test_label = torch.squeeze(test_label)
            h = h.data

            y_pred, h = model(test_data.to(device).float(), h)
            y_pred = torch.squeeze(y_pred)
 
            pred.append(y_pred.float().cpu().detach().numpy()[:,0])
            label.append(test_label.cpu().detach().numpy()[:,0])


            # Compute loss and metrics
            loss = criterion(y_pred.float(), test_label.to(device).float())
            m = nn.L1Loss()
            mae_val.append(
                m(y_pred.float()[:,0], test_label.to(device).float()[:,0]).item()
            )

            mae_std.append(
                m(y_pred.float()[:,1], test_label.to(device).float()[:,1]).item()
            )

            r = nn.MSELoss()
            rmse_val.append(
               np.sqrt( r(y_pred.float()[:,0], test_label.to(device).float()[:,0]).item())
            )

            rmse_std.append(
               np.sqrt( r(y_pred.float()[:,1], test_label.to(device).float()[:,1]).item())
            )

            loss_1 = criterion(y_pred.float()[:,0], train_label.to(device).float()[:,0])
            loss_2 = criterion(y_pred.float()[:,1], train_label.to(device).float()[:,1])
            loss = loss_1 + beta*loss_2
            losses.append(loss.item())


        mae_1 = np.mean((mae_val))
        mae_2 = np.mean((mae_std))
        rmse_1 = np.mean((rmse_val))
        rmse_2 = np.mean((rmse_std))

        test_loss = np.mean(losses)   
        print("Epoch: {} Train loss: {:.2f}, Dev loss: {:.2f}, Test loss: {:.2f}, Test MAE : {:.2f} mmHg, Test MAE (std) : {:.2f} mmHg, Test RMSE: {:.2f} mmHg, Std test RMSE: {:.2f} mmHg".format(
            epoch, train_loss, dev_loss, test_loss, mae_1, mae_2, rmse_1, rmse_2))
        ep_dev_loss.append(dev_loss)
        ep_train_loss.append(train_loss)
   
    ##Plot to display predicted vs actual values
    # false = np.concatenate(pred)
    # true = np.concatenate(label)
    # plt.figure()
    # fig, ax = plt.subplots()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # print(len(false))
    # plt.plot(range(len(false)), false, color = 'red')
    # plt.plot(range(len(true)), true)
    # ax.legend(['Predicted values', 'Actual values'], fontsize = 12)
    # plt.xlabel('Test patient', fontsize = 10)
    # plt.ylabel('ABPd (mmHg)', fontsize = 10)

    ##Plot to display the learning curve
    # fig, ax = plt.subplots()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.plot(range(epoch), ep_train_loss)
    # plt.plot(range(epoch), ep_dev_loss)
    # plt.title('Learning curve - {} - W = {}'.format(task, W))
    # plt.xlabel('Epoch')
    # plt.ylabel('MSE')
    # plt.legend(['Training Loss', 'Validation Loss'])
    # plt.savefig(r"C:\Users\USER\OneDrive\Summer_project\Pics - Copy\LC_{}_{}_{}_{}_{}".format(task, severe, hidden_dim, n_layers, EPOCHS))
    # if save:
    #   model_path = r'C:\Users\USER\OneDrive\Summer_project\Azure\models\GRU_{}_{}_{}_{}_{}'.format(task, severe, hidden_dim, n_layers, EPOCHS)
    #   torch.save(model, model_path)

    return model

if task == 'pressure_experiment':
    #This task corresponds to the removal of correlated Blood Pressures, with transfer learning, varying time window
    #Prediction of std 
    #Data preprocessing
    data, labels = pr.exp_pr(lb, transfer=False, window = W)
    data_TBI, labels_TBI = pr_TBI.exp_pr(lb, transfer=False, window = W)
    final_data_TBI = np.array(data_TBI)
    final_data_TBI = np.transpose(final_data_TBI, (0,2,1))
    data = np.transpose(data, (0,2,1))

    #labels
    labels_2 = np.array(labels.to_numpy())
    labels_TBI = np.array(labels_TBI.to_numpy())

    #Dataset split 
    X_train, X_test, y_train, y_test = train_test_split(data, labels_2, test_size=0.2, shuffle = True, random_state=random_state)
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    #Load data in batches 
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    dev_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size, drop_last=True)

    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

    print('Pretraining with the whole dataset')
    pretrained_model = train(train_loader, dev_loader, test_loader, learn_rate = lr, hidden=128, layers=5, task = lb, save = True, model_type="GRU", EPOCHS = n_epochs, severe = '', output_dim = 2)

    print('Finetuning for TBI cohort')
    X_train, X_test, y_train, y_test = train_test_split(final_data_TBI, labels_TBI, test_size=0.2, shuffle = True, random_state=random_state)
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    dev_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size, drop_last=True)

    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
    pretrained_model = train(train_loader, dev_loader, test_loader, learn_rate = lr_finetuning, hidden=128, layers= 5, task = lb, save = True, model_type=pretrained_model, EPOCHS = n_epochs_finetuning, severe = '', output_dim = 2)

if task == 'std_augmented':
    #This task corresponds to std prediction with transfer learning, on a varying time window 
    data, labels = pr.std_pr(lb, transfer=False, window=W)
    data_TBI, labels_TBI = pr_TBI.std_pr(lb, transfer=False, window=W)
    final_data_TBI = np.array(data_TBI)
    final_data_TBI = np.transpose(final_data_TBI, (0,2,1))
    data = np.transpose(data, (0,2,1))
    labels_2 = np.array(labels.to_numpy())
    labels_TBI = np.array(labels_TBI)

    #Dataset split
    X_train, X_test, y_train, y_test = train_test_split(data, labels_2, test_size=0.2, shuffle = True, random_state=random_state)
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    #Load data in batches 
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    dev_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size, drop_last=True)

    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

    print('Pretraining with the whole dataset')
    pretrained_model = train(train_loader, dev_loader, test_loader, learn_rate = lr,  task = lb, save = True, model_type='GRU', EPOCHS = n_epochs, severe = '', output_dim = 2)

    print('Finetuning for TBI cohort')
    X_train, X_test, y_train, y_test = train_test_split(final_data_TBI, labels_TBI, test_size=0.2, shuffle = True, random_state=random_state)
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    dev_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size, drop_last=True)

    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
    pretrained_model = train(train_loader, dev_loader, test_loader, learn_rate = lr_finetuning,  task = lb, save = True, model_type=pretrained_model, EPOCHS = n_epochs_finetuning, severe = '', output_dim = 2)


elif task == 'VAE_augmentation':
    #This task corresponds to VAE data augmentation. The VAE model we trained is used. 

    data, labels = pr.time_series_pr(lb, transfer=False)
    data_TBI, labels_TBI = pr_TBI.time_series_pr(lb, transfer=False)

    final_data = np.array(data)
    final_data = np.transpose(final_data, (0,2,1))

    #Extra preprocessing for generation 
    test_labels = np.transpose(np.array([[labels]*24]), (2,0,1))
    concatenated_data = np.concatenate((final_data, test_labels), 1)

    ##VAE training 
    # da = DataAugmentation(True, concatenated_data, random_state)
    # da.augment_VAE(50,16,0.003)
    # da = DataAugmentation(True, concatenated_data, random_state)
    # vae = VAE(concatenated_data.shape[2]*concatenated_data.shape[1])
    # model = da.train_VAE(vae, 400, 16, 0.003)

    # with open('VAE_model.pkl', 'wb') as outp:
    #     pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

    #Load the VAE model we trained 
    with open('VAE_model.pkl', 'rb') as inp:
        model = pickle.load(inp)

    n_samples = 300 #Number of samples to be generated
    latent_dim = 20
    z = torch.randn(n_samples, latent_dim).to(device)
    with torch.no_grad():
        z = z.to(device).double()
        samples = model.to(device).decode(z)
        samples = samples.cpu()

    new_samples = samples.reshape((n_samples, concatenated_data.shape[1], -1)).detach().numpy()[:,0:-1,:]
    new_labels = new_samples[:,-1,:]
    new_labels = new_labels[:,0]

    #Augmented data with newly generated samples
    final_data = np.concatenate((new_samples, final_data))
    labels = np.concatenate((new_labels, labels))
    final_data_TBI = np.array(data_TBI)
    final_data_TBI = np.transpose(final_data_TBI, (0,2,1))

    print('Pretraining with the augmented dataset')
    #Pretraining with the whole dataset 
    X_train, X_test, y_train, y_test = train_test_split(final_data, labels, test_size=0.2, shuffle = True, random_state=random_state)
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    dev_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size, drop_last=True)

    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
    pretrained_model = train(train_loader, dev_loader, test_loader, learn_rate = lr, hidden=128, layers= 5, task = lb, save = True, model_type="GRU", EPOCHS = n_epochs, severe = '')

    print('Finetuning for TBI cohort')
    X_train, X_test, y_train, y_test = train_test_split(final_data_TBI, labels_TBI, test_size=0.2, shuffle = True, random_state=random_state)
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    dev_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size, drop_last=True)

    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
    pretrained_model = train(train_loader, dev_loader, test_loader, learn_rate = lr_finetuning, hidden=128, layers= 5, task = lb, save = True, model_type=pretrained_model, EPOCHS = n_epochs_finetuning, severe = '')
