from VAE import VAE 
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
is_cuda = torch.cuda.is_available()
import numpy as np

class DataAugmentation:
    def __init__(self, train, ds, random_state, model = None):
        self.train = train 
        self.ds = ds
        self.random_state = random_state
        if not self.train:
            self.model = model

    def train_VAE(self, model, num_epochs, batch_size, learning_rate):
        X_train, X_test = train_test_split(self.ds, test_size=0.2, shuffle = True, random_state=self.random_state)
        train_data = torch.from_numpy(X_train)
        loader_train = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
        test_data = torch.from_numpy(X_test)
        loader_test = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


        if is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print('Device', device)  
        train_losses = []
        test_losses = []
        model = model.double()
        for epoch in range(num_epochs):     
            total_loss, total_NLL_loss, total_KLD_loss = 0, 0, 0
            model.train()
            for data in loader_train:
                data = data.double()
                data = data.view((batch_size,-1))

                data = data.to(device=device)

                recon_x, mu, logvar = model.forward(data)

                loss, NLL_loss, KLD_loss = model.loss_function_VAE(recon_x, data, mu, logvar)
                total_loss += loss.item()
                total_NLL_loss += NLL_loss.item()
                total_KLD_loss += KLD_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_losses.append([total_loss / len(loader_train.dataset), total_NLL_loss / len(loader_train.dataset), total_KLD_loss / len(loader_train.dataset)])

            total_loss, total_NLL_loss, total_KLD_loss = 0, 0, 0
            model.eval()
         
            with torch.no_grad():
                for data in (loader_test):
                    data = data.view((batch_size,-1))

                    data = data.double()
                    data = data.to(device=device)
                    recon_x, mu, logvar = model.forward(data)
                    loss, NLL_loss, KLD_loss = model.loss_function_VAE(recon_x, data, mu, logvar, beta=5)
                    total_loss += loss.item()
                    total_NLL_loss += NLL_loss.item()
                    total_KLD_loss += KLD_loss.item()

            test_losses.append([total_loss / len(loader_test.dataset), total_NLL_loss / len(loader_test.dataset), total_KLD_loss / len(loader_test.dataset)])
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch: {epoch}, train loss = {train_losses[-1][0]:.2f}, test loss = {test_losses[-1][0]:.2f}, lr = {lr:.5f}')
        # save the model
        print(np.array(test_losses).shape)
        if epoch == num_epochs - 1:
            with torch.no_grad():
                torch.jit.save(torch.jit.trace(model, (data), check_trace=False),
                    'VAE_model.pth')
            test_losses = np.array(test_losses)
            train_losses = np.array(train_losses)
            plt.figure()
            plt.plot(range(num_epochs), train_losses[:,0])
            plt.plot(range(num_epochs), test_losses[:,0])
            plt.title('VAE Learning curve - Total loss{}')
            plt.xlabel('Epoch')
            plt.ylabel('Total loss')
            plt.legend(['Training Loss', 'Validation Loss'])

            plt.figure()
            plt.plot(range(num_epochs), train_losses[:,1])
            plt.plot(range(num_epochs), test_losses[:,1])
            plt.title('VAE Learning curve - MSE loss{}')
            plt.xlabel('Epoch')
            plt.ylabel('MSE loss')
            plt.legend(['Training Loss', 'Validation Loss'])

            plt.show()

            plt.figure()
            plt.plot(range(num_epochs), train_losses[:,2])
            plt.plot(range(num_epochs), test_losses[:,2])
            plt.title('VAE Learning curve - KL loss{}')
            plt.xlabel('Epoch')
            plt.ylabel('KL loss')
            plt.legend(['Training Loss', 'Validation Loss'])

            plt.show()

    def augment_VAE(self, num_epochs, batch_size, lr):
        self.model = VAE(self.ds.shape[2]*self.ds.shape[1])
        self.train_VAE(self.model, num_epochs, batch_size, lr)
        
        
        