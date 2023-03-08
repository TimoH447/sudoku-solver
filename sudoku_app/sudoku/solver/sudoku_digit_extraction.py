import json

import torch
from torch import nn 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor,ToPILImage
from matplotlib import pyplot as plt
from PIL import Image

import sudoku_cv

ASSET_DIR = "/Users/timoh/OneDrive/Dokumente/Bildung/Programmieren/old_version/assets/"

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

training_data = datasets.MNIST(
    root=ASSET_DIR+"data",
    train=True,
    download= True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root=ASSET_DIR+"data",
    train=False,
    download= True,
    transform=ToTensor(),
)



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def make_model(training_data,test_data):
    batch_size = 64

   
    #Create data loaders
    train_dataloader = DataLoader(training_data,batch_size=batch_size)
    test_dataloader = DataLoader(test_data,batch_size=batch_size)
    
    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), ASSET_DIR+ "model.pth")
    print("Saved PyTorch Model State to model.pth")

def predict():
    #make_model(training_data,test_data)

    model = NeuralNetwork()
    model.load_state_dict(torch.load(ASSET_DIR+"model.pth"))

    classes = ["0","1","2","3","4","5","6","7","8","9"]
    #model.eval()
    x, y = test_data[0][0], test_data[0][1]
    print(test_data[0][0].shape)
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

    plt.imshow(ToPILImage()(test_data[0][0]))
    plt.show()

def predict_digit(image):
    model = NeuralNetwork()
    model.load_state_dict(torch.load(ASSET_DIR+"mnist-b07bb66b.pth", map_location=torch.device('cpu')))
    

    classes = ["0","1","2","3","4","5","6","7","8","9"]
    #model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485],
                         std=[0.229])
                         ])
    input_tensor = transform(image)

    # Apply the transformation pipeline to the image

    input_tensor = input_tensor.view(1,1,28,28)
    
    with torch.no_grad():
        output = model(input_tensor)

    # Convert the output to a human-readable format
    print(output)
    prediction = classes[output[0].argmax(0)]
    print(prediction)

def digit_images_from_sudoku(image_path):
    sudoku = sudoku_cv.extract_sudoku(image_path=ASSET_DIR+"Sudoku_front.jpg",saving=False)

    digits = []
    for i in range(9):
        for j in range(9):
            digit_box_coordinates = (j*30,i*30,j*30+30,i*30+30)
            digits.append(sudoku.crop(digit_box_coordinates))
    
    return digits

def directional_analysis(digit_img):
    digit_img = digit_img.crop((3,3,25,25))
    np_image = sudoku_cv.pillow2np(digit_img)
    np_image = sudoku_cv.draw_image_border(np_image)
    
    start = sudoku_cv.get_point_on_contour(np_image)
    if start is None:
        return []
    x,y=start
    
    direction = 0 # 0: right, 1: right-down, 2: down ...
    directions_tracing =[]
    coordinates = []

    while True:

        x,y,direction = sudoku_cv.next_border_clockwise(np_image,x,y,direction)
        directions_tracing.append(direction)
        coordinates.append((x,y))

        if (x,y)==start:
            break
    
    return directions_tracing
def analyse_sudoku_digits(digits):
    analysis= [ ]
    for digit in digits:
        i=0
        directions = directional_analysis(digit)
        analysis.append(directions)

    with open(ASSET_DIR+"analysis.json","w+") as file:
        json.dump(analysis,file)

def digit_preprocess(digit_img):
    # crop to 28x28 pixels
    digit_img = digit_img.crop((1,1,29,29))
    digit_img = sudoku_cv.grayscale_img(digit_img)
    digit_img = sudoku_cv.adaptive_thresholding(digit_img)
    return digit_img

if __name__=="__main__":
    #sudoku_cv.extract_sudoku(ASSET_DIR + "Sudoku_front.jpg")
    pass    
