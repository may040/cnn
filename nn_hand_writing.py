import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from io import open
from PIL import Image
import pathlib
from torch.autograd import Variable


#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
print(f"Using {device} device")


training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
)


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


learning_rate = 1e-3
batch_size = 64
epochs = 1
loss_fn = nn.CrossEntropyLoss()


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 10, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 30, 5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(480, 60),
            nn.ReLU(),
            nn.Linear(60, 10)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

if os.path.isfile('cnn_hand_writing.pt'):
    model = torch.load('cnn_hand_writing.pt')


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):

        X = X.to(device)
        y = y.to(device)

        pred = model(X).to(device)
        loss = loss_fn(pred, y)

        # Bpg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    n = 0
    with torch.no_grad():
        for X, y in dataloader:

            X = X.to(device)
            y = y.to(device)
            print(X.size())
            pred = model(X).to(device)
            if n == 0:
                differenceIndices = [i for i, (x, y) in enumerate(
                    zip(pred.argmax(1), y)) if x != y]
                _printRandomWrongPred(pred.argmax(1), y, differenceIndices)

                n += 1

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def _printRandomWrongPred(prediction, labels, differenceIndices):

    labels_map = {
        0: "Zero",
        1: "One",
        2: "Two",
        3: "Three",
        4: "Four",
        5: "Five",
        6: "Six",
        7: "Seven",
        8: "Eight",
        9: "Nine",
    }

    figure = plt.figure(figsize=(8, 8))
    for i in range(0, len(differenceIndices)):
        features, label = next(iter(test_dataloader))
        matchesIndices = ((label.to("cuda") == labels[differenceIndices[i]].to(
            "cuda")).nonzero(as_tuple=True)[0])
        img = features[matchesIndices[0]]
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title("Predcition: {} and original: {}".format(
            prediction[differenceIndices[i]], labels[differenceIndices[i]]))
        plt.show()


def _classifyImage(img_path):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((28, 28), transforms.InterpolationMode.BICUBIC), transforms.Normalize((0.1307,), (0.3081,))])

    image = Image.open(img_path)
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)
    input = Variable(img_tensor)
    output = model(input)
    plt.imshow(image)
    plt.title("Predcition: {} and original: {}".format(
            output.argmax(1).item(),5))
    plt.show()


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    if t==epochs-1:
        test_loop(test_dataloader, model, loss_fn)

torch.save(model, 'cnn_hand_writing.pt')

_classifyImage(r'path')

print("Done!")
