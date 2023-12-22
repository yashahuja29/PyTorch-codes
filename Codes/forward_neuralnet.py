import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"

input_size = 784
hidden_size = 100
num_classes = 10
num_epoch = 11
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root = './data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root = './data',train=False,transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size= batch_size,shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size= batch_size,shuffle = False)

example = iter(train_loader)
sample, label = next(example)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(sample[i][0],cmap="gray")
# plt.show()
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
model = NeuralNet(input_size,hidden_size,num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimisor = torch.optim.Adam(model.parameters(),lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epoch):
    for i,(images,labels) in enumerate(train_loader):

        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)
        optimisor.zero_grad()
        loss.backward()
        optimisor.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epoch}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')


with torch.no_grad():
    n_correct = 0
    n_sample = 0
    for images, labels in test_loader:
        images =images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs,1)
        n_sample += label.shape[0]
        n_correct += (predictions==labels).sum().item()
    
    acc = 100.0 * n_correct/n_sample
    print(f'accuracy = {acc:.4f}')

