import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn   
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim


transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])


data_dir = './Data_10'
train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()

        # Блок 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm2d(12)

        # Блок 2
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, padding=1)
        self.bn2 = nn.BatchNorm2d(24)

        # Пулинг
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Блок 3
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(48)

        # Блок 4
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # Полносвязный слой 
        self.fc = nn.Linear(3136, 10)

    def forward(self, inp):
        out = F.relu(self.bn1(self.conv1(inp)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)

        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = self.pool(out)

        out = out.view(out.size(0), -1)  
        return self.fc(out)



model = ImageModel()



def test_accuracy():
    model.eval()

    accuracy = 0 
    total = 0     

    for test_data in test_loader:
        images, labels = test_data  
        output = model(images)      
        predict = torch.max(output.data, 1)[1]

        print(predict)

        accuracy += (predict == labels).sum().item()
        total += labels.size(0)  


    print(accuracy, total)  

    if total == 0:
        return 0
    return 100 * accuracy / total


train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
test_loader = DataLoader(test_set, batch_size=10, shuffle=False)


loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
num_epochs = 3


best_accuracy = 0.0
model_save_path = './LearnModel.pth'

# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):  # loader должен быть train_loader
#         model.train()
#         optimizer.zero_grad()              # очищаем градиенты
#         output = model(images)            # прямое распространение
#         error = loss(output, labels)      # считаем ошибку
#         error.backward()                  # обратное распространение
#         optimizer.step()                  # обновление весов


#     # Проверка точности после каждой эпохи
#     accuracy = test_accuracy()
#     print('Epoch: %d; Accuracy: %.2f%%' % (epoch + 1, accuracy))

#     if accuracy > best_accuracy:
#          best_accuracy = accuracy
#          torch.save(model.state_dict(), model_save_path)



load_model = ImageModel()
load_model.load_state_dict(torch.load(model_save_path))
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

images, true_labels =  next(iter(test_loader))

print('True labels: ', end='') 
for i in range(10): 
    print(classes[true_labels[i]], end=' ') 
print()  

output = load_model(images) 
predict = torch.max(output, 1)[1]
print('Predicted: ', end='')


for i in range(10):
    print(classes[predict[i]], end=' ') 
print()

# объединяем все 10 изображений в одно
images = torchvision.utils.make_grid(images)

# мы выполняли нормализацию, и значения изображений находятся в диапазоне от -1 до 1,
# но для вывода нужен диапазон от 0 до 1
images = images / 2 + 0.5

# функция для рисования на plot изображения. Изображение предварительно нужно преобразовать
# для этого транспонируем изображение как numpy массив
plt.imshow(np.transpose(images.numpy(), (1, 2, 0)))
plt.show()
