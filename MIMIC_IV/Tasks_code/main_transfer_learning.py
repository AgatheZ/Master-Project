import numpy as np
import torch
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
import GRU
from VAE import VAE
import pandas as pd 
import torch.nn as nn
from preprocessing import Preprocessing
from torch.utils.data import TensorDataset, DataLoader
import warnings
import torch.utils.data as utils
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from data_augmentation import DataAugmentation
warnings.filterwarnings("ignore")
import pickle
time.clock = time.time
import sys

#Parameters
nb_hours = 24
random_state = 1
TBI_split = False
tuning = False
SHAP = False
imputation = 'carry_forward'
model_name = 'Stacking'
lr = 0.001
learning_rate_decay = 7 
n_epochs = 10
batch_size = 16
lb = 'ABPd'
is_cuda = torch.cuda.is_available()
task = 'augmentation' #augmentation or cohort split



# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('Device', device)  
def train(train_loader, dev_loader, test_loader, learn_rate, save = True, task = 'ABPd', hidden=512, layers= 49, EPOCHS=5, model_type="GRU", severe = '', output_dim = 1):
    
    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]
    print(output_dim)
    hidden_dim = hidden
    n_layers = layers
    # Instantiating the models
    if model_type == "GRU":
        model = GRU.GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    else:
        model = model_type
    model.to(device)
    
    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    ep_train_loss =[]
    ep_dev_loss = []
    # Start training loop
    for epoch in range(1,EPOCHS+1):
        start_time = time.clock()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        losses, mae = [], []
        label, pred = [], []
        epoch_losses = []
        y_pred_col = []
        
        model.train()
        for x, train_label in train_loader:
            counter += 1
            h = h.data
            print('train')
            model.zero_grad()
            # x = x + (0.1**0.5)*torch.randn(x.shape)
            y_pred, h = model(x.to(device).float(), h)
            y_pred = torch.squeeze(y_pred)
            y_pred_col.append(y_pred)

            pred.append(y_pred.cpu().detach().numpy())
            label.append(train_label)
            loss = criterion(y_pred.float(), train_label.to(device).float())
            m = nn.L1Loss()
            mae.append(
                
                    (m(y_pred.float(), train_label.to(device).float())).item()
            )
            losses.append(loss.item())

            #backward pass 
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
        
        train_mae = np.mean((mae))
        train_loss = np.mean(losses)
        train_pred_out = pred
        train_label_out = label
        model.eval()
        
        #validation set 
        losses, mae = [], []
        label, pred = [], []
        for dev_data, dev_label in dev_loader:
            # Forward pass : Compute predicted y by passing train data to the model
            h = h.data
            y_pred, h = model(dev_data.to(device).float(), h)
            y_pred = torch.squeeze(y_pred)
            print('dev')
            # Save predict and label
            pred.append(y_pred)
            label.append(dev_label)

            # Compute loss
            loss = criterion(y_pred.float(), dev_label.to(device).float())
            m = nn.L1Loss()
            mae.append(m(y_pred.float(), dev_label.to(device).float()).item())
            losses.append(loss.item())
            
        dev_mae = np.mean((mae))
        dev_loss = np.mean(losses)
        dev_pred_out = pred
        dev_label_out = label
        current_time = time.clock()

        losses, mae = [], []
        label, pred = [], []
        model.eval()

        for test_data, test_label in test_loader:
            test_data = torch.squeeze(test_data)
            test_label = torch.squeeze(test_label)
            
            # Forward pass : Compute predicted y by passing train data to the model
            h = h.data
            print(h.shape)
            print(test_data.shape)
            y_pred, h = model(test_data.to(device).float(), h)
            y_pred = torch.squeeze(y_pred)
            
            # Save predict and label
            pred.append(y_pred.cpu().detach().numpy())
            label.append(test_label.cpu().detach().numpy())

            # Compute loss
            loss = criterion(y_pred.float(), test_label.to(device).float())
            m = nn.L1Loss()
            mae.append(
                m(y_pred.float(), test_label.to(device).float()).item()
            )
            losses.append(loss.item())
            
        test_mae = np.mean((mae))
        test_loss = np.mean(losses)
        test_pred_out = pred
        test_label_out = label
                
        epoch_losses.append([
             train_loss, dev_loss, test_loss,
             train_mae, dev_mae, test_mae,
             train_pred_out, dev_pred_out, test_pred_out,
             train_label_out, dev_label_out, test_label_out,
         ])
        
        pred = np.asarray(pred)
        label = np.asarray(label)
        
        mae_1 = mean_absolute_error(label[0], pred[0])
        mae_2 = mean_absolute_error(label[1], pred[1])
        rmse_1 = np.sqrt(mean_squared_error(label[0], pred[0]))
        rmse_2 = np.sqrt(mean_squared_error(label[1], pred[1]))
        print("Epoch: {} Train loss: {:.4f}, Dev loss: {:.4f}, Test loss: {:.4f}, Test MAE : {:.4f}, Test MAE (std) : {:.4f}".format(
            epoch, train_loss, dev_loss, test_loss, mae_1, mae_2))
        print("Epoch: {} Test RMSE: {:.4f}, Std test RMSE: {:.4f}".format(
            epoch, rmse_1, rmse_2))
        ep_dev_loss.append(dev_loss)
        ep_train_loss.append(train_loss)
    
    plt.figure()
    plt.plot(range(epoch), ep_train_loss)
    plt.plot(range(epoch), ep_dev_loss)
    plt.title('Learning curve - {}'.format(task))
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.savefig(r"C:\Users\USER\OneDrive\Summer_project\Pics - Copy\LC_{}_{}_{}_{}_{}".format(task, severe, hidden_dim, n_layers, EPOCHS))
    if save:
      model_path = r'C:\Users\USER\OneDrive\Summer_project\Azure\models\GRU_{}_{}_{}_{}_{}'.format(task, severe, hidden_dim, n_layers, EPOCHS)
      torch.save(model, model_path)
    return model




##data loading 
df_24h = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_24hour.csv', delimiter=',')
df_48h = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_48hour.csv', delimiter=',')
df_med = pd.read_csv(r"C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_med.csv", delimiter=',')
df_demographic_augmented = pd.read_csv(r"C:\Users\USER\OneDrive\Summer_project\Azure\data\demographics_mimic4_augmented.csv", delimiter=',')
df_demographic = pd.read_csv(r"C:\Users\USER\OneDrive\Summer_project\Azure\data\demographics_mimic4.csv", delimiter=',')

if task in ['std', 'std_augmented']:
    df_hourly = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_hour_std.csv', delimiter=',')
    df_hourly_augmented = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_hour_augmented_std.csv', delimiter=',')
else: 
    df_hourly = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_hour.csv', delimiter=',')
    df_hourly_augmented = pd.read_csv(r'C:\Users\USER\OneDrive\Summer_project\Azure\data\preprocessed_mimic4_hour_augmented.csv', delimiter=',')

# features = pd.read_csv(r'MIMIC_IV\resources\features_reg.csv', header = None)
# features = features.loc[:416,1] 

pr = Preprocessing(df_hourly_augmented, df_24h, df_48h, df_med, df_demographic_augmented, nb_hours, TBI_split, random_state, imputation)
pr_TBI = Preprocessing(df_hourly, df_24h, df_48h, df_med, df_demographic, nb_hours, TBI_split, random_state, imputation)



###############################################################################
if task == 'std_augmented':
    data, labels = pr.std_pr(lb, transfer=False)
    data_TBI, labels_TBI = pr_TBI.std_pr(lb, transfer=False)
    
    final_data_TBI = np.array(data_TBI)
    final_data_TBI = np.transpose(final_data_TBI, (0,2,1))
    data = np.transpose(data, (0,2,1))
    
    labels_2 = np.array(labels.to_numpy())
    X_train, X_test, y_train, y_test = train_test_split(data, labels_2, test_size=0.2, shuffle = True, random_state=random_state)
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)


    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    dev_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size, drop_last=True)

    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

    print('Pretraining with the whole dataset')
    pretrained_model = train(train_loader, dev_loader, test_loader, learn_rate = lr, hidden=512, layers=49, task = lb, save = True, model_type="GRU", EPOCHS = n_epochs, severe = '', output_dim = 2)

    print('Finetuning for TBI cohort')
    X_train, X_test, y_train, y_test = train_test_split(final_data_TBI, labels_TBI, test_size=0.2, shuffle = True, random_state=random_state)
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    dev_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size, drop_last=True)

    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
    pretrained_model = train(train_loader, dev_loader, test_loader, learn_rate = lr, hidden=512, layers= 49, task = lb, save = True, model_type="GRU", EPOCHS = n_epochs, severe = '')

if task == 'std':

    data, labels = pr.std_pr(lb, transfer=False)
    data = np.transpose(data, (0,2,1))
    print(data.shape)
    labels_2 = np.array(labels.to_numpy())
    X_train, X_test, y_train, y_test = train_test_split(data, labels_2, test_size=0.2, shuffle = True, random_state=random_state)
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)


    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    dev_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size, drop_last=True)

    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
    pretrained_model = train(train_loader, dev_loader, test_loader, learn_rate = lr, hidden=16, layers=1, task = lb, save = True, model_type="GRU", EPOCHS = n_epochs, severe = '', output_dim = 2)

elif task == 'augmentation':

    # data, labels, data_mild, labels_mild, data_severe, labels_severe = pr.time_series_pr(lb, transfer=True)

    data, labels = pr.time_series_pr(lb, transfer=False)
    data_TBI, labels_TBI = pr_TBI.time_series_pr(lb, transfer=False)
    test_labels = np.transpose(np.array([[labels]*24]), (2,0,1))

    final_data = np.array(data)
    final_data = np.transpose(final_data, (0,2,1))

    concatenated_data = np.concatenate((final_data, test_labels), 1)

    da = DataAugmentation(True, concatenated_data, random_state)
    # da.augment_VAE(50,16,0.003)

    # da = DataAugmentation(True, concatenated_data, random_state)
    # vae = VAE(concatenated_data.shape[2]*concatenated_data.shape[1])
    # model = da.train_VAE(vae, 400, 16, 0.003)


    # with open('VAE_model.pkl', 'wb') as outp:
    #     pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

    with open('MIMIC_IV\Tasks_code\VAE_model.pkl', 'rb') as inp:
        model = pickle.load(inp)

    da.show_reconstruct(16, model)
    n_samples = 1000
    latent_dim = 20
    z = torch.randn(n_samples, latent_dim).to(device)
    with torch.no_grad():
        z = z.to(device).double()
        samples = model.decode(z)
        samples = samples.cpu()

    new_samples = samples.reshape((n_samples, concatenated_data.shape[1], -1)).detach().numpy()[:,0:-1,:]
    print(new_samples.shape)
    new_labels = new_samples[:,-1,:]
    new_labels = new_labels[:,0]
    print(new_samples.shape)
    plt.figure()
    plt.plot(range(24), new_samples[12][3])
    plt.show()
    np.save('new_samples_vae.npy', new_samples)
    final_data = np.concatenate((new_samples, final_data))
    print(final_data.shape)
    labels = np.concatenate((new_labels, labels))
    final_data_TBI = np.array(data_TBI)
    final_data_TBI = np.transpose(final_data_TBI, (0,2,1))
    # np.save('final_data_TBI.npy', final_data_TBI)
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
    pretrained_model = train(train_loader, dev_loader, test_loader, learn_rate = lr, hidden=512, layers= 49, task = lb, save = True, model_type="GRU", EPOCHS = n_epochs, severe = '')

    print('Finetuning for TBI cohort')
    X_train, X_test, y_train, y_test = train_test_split(final_data_TBI, labels_TBI, test_size=0.2, shuffle = True, random_state=random_state)
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    dev_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size, drop_last=True)

    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
    pretrained_model = train(train_loader, dev_loader, test_loader, learn_rate = lr, hidden=512, layers= 49, task = lb, save = True, model_type="GRU", EPOCHS = n_epochs, severe = '')

else: 
    print('Pretraining with the whole dataset')
    #Pretraining with the whole dataset 
    X_train, X_test, y_train, y_test = train_test_split(final_data, labels, test_size=0.2, shuffle = True, random_state=random_state)
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    dev_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size, drop_last=True)

    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
    pretrained_model = train(train_loader, dev_loader, test_loader, learn_rate = lr, hidden=512, layers= 49, task = lb, save = True, model_type="GRU", EPOCHS = n_epochs, severe = '')

    print('Finetuning for mild cohort')
    #Fine-tuning tasks - mild

    final_data = np.array(data_mild)
    final_data = np.transpose(final_data, (0,2,1))

    X_train, X_test, y_train, y_test = train_test_split(final_data, labels_mild, test_size=0.2, shuffle = True, random_state=random_state)
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    dev_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size, drop_last=True)

    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

    # PATH = r'C:\Users\USER\OneDrive\Summer_project\Azure\Master-Project\MIMIC_IV\models\GRU_{}_{}_{}_{}_{}'.format(task, '', 512, 49, 25) 
    # pretrained_model = torch.load(PATH)


    mild_model = train(train_loader, dev_loader, test_loader, learn_rate = lr, task = lb,hidden=512, layers= 49,  save = True, model_type=pretrained_model, EPOCHS = n_epochs, severe= 'mild')

    print('Finetuning for severe cohort')
    #Fine-tuning tasks - severe

    final_data = np.array(data_severe)
    final_data = np.transpose(final_data, (0,2,1))
    X_train, X_test, y_train, y_test = train_test_split(final_data, labels_severe, test_size=0.2, shuffle = True, random_state=random_state)
    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    dev_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size, drop_last=True)

    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

    severe_model = train(train_loader, dev_loader, test_loader, learn_rate = lr, task = lb, hidden=512, layers= 49, save = True, model_type=pretrained_model, EPOCHS = n_epochs, severe = 'severe')
