import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from model import CSIHAR
from data import CSIDataset
import torch.optim as optim


bs = 16
lr = 1e-3
epochs = 500
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


if __name__ == "__main__":
    best_training_loss = 9999.99
    best_testing_acc = 0

    train_dataset = CSIDataset(metadata="train.txt")
    test_dataset = CSIDataset(metadata="test.txt")

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=False
    )

    model = CSIHAR(in_channels=1, num_class=10, num_domains=9)
    model = model.to(device)

    #criterion = nn.CrossEntropyLoss()
    
    weights = torch.tensor([24.64864864864865, 52.114285714285714, 26.057142857142857, 12.493150684931507, \
                            12.579310344827586, 50.66666666666667, 29.901639344262296, 41.45454545454546,\
                            53.64705882352941, 1.5470737913486003]).to(device)
  
    criterion = nn.CrossEntropyLoss(weight=weights)
    criterion_domain = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels, domains = data
            
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            domains = domains.to(device, dtype=torch.long)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs, domain_outputs = model(inputs)
            loss_cls = criterion(outputs, labels)
            loss_domain = criterion_domain(domain_outputs, domains)

            loss = loss_cls + 0.5 * loss_domain

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:    # print every 500 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i+1):.3f}')
            
        
        training_loss = running_loss / (i+1)
        print(f'Finished Training for epoch {epoch}, loss: {training_loss}')
        if training_loss < best_training_loss:
            best_training_loss = training_loss
            model_path = 'train_{}'.format(epoch)
            torch.save(model.state_dict(), model_path)
            print("Model Saved")
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, domains = data
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)
                
                outputs, domain_output = model(inputs)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            acc = 100 * correct // total
            print(f'Accuracy of the network on the test data: {acc} %')
