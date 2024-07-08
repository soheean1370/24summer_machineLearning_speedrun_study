# 데이터 작업하기 
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 공개 데이터셋에서 학습 데이터를 내려받는다
training_data = datasets.FashionMNIST(
    root ="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 공개 데이터셋에서 테스트 데이터를 내려받는다
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64 # 배치크기를 64로 정의

#데이터로더를 생성
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
# 데이터셋의 구조와 형태를 확인
for X,y in test_dataloader:
    print(f"Shape of X[N, C, H, W]: {X.shape}") 
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
#모델 만들기
#학습에 사용할 cpu나 gpu,mps 장치를 얻는다
device = (
    "cuda"
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 28x28 크기의 이미지 데이터를 입력으로 받아 10개의 클래스로 분류하는 인공신경망을 정의
class NeuralNetwork(nn.Module):
    def __init__(self): # 신경망 계층 정의
        super().__init__()
        self.flatten = nn.Flatten() # 입력 이미지를 일렬로 펴는 역할
        self.linear_relu_stack = nn.Sequential( # 여러 계층을 순차적으로 쌓은 신경망 정의
            nn.Linear(28*28, 512), # 28*28차원을 512차원으로
            nn.ReLU(), # 활성화 함수
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self,x): # 신경망 데이터 전달 방법
        x = self.flatten(x) # x를 일렬로 편다
        logits = self.linear_relu_stack(x) #연산 수행
        return logits # 모델의 출력값 반환.(각 클래스에 대한 점수)
    
model = NeuralNetwork().to(device)
print(model)

#모델 매개변수 최적화하기
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 데이터셋의 전체 크기를 가져옴
    for batch, (X, y) in enumerate(dataloader):  # 데이터 로더에서 배치 단위로 데이터를 반복
        X, y = X.to(device), y.to(device)  # 입력 데이터(X)와 레이블(y)을 mps로 이동

        #예측오류계산
        pred = model(X)  # 모델을 사용하여 예측값을 계산
        loss = loss_fn(pred, y)  # 예측값과 실제값 사이의 손실(loss)을 계산

        #역전파
        loss.backward()  # 역전파를 통해 모델의 가중치를 조정
        optimizer.step()  # 옵티마이저를 사용하여 가중치를 업데이트
        optimizer.zero_grad()  # 옵티마이저의 기울기(gradient)를 초기화

        if batch % 100 == 0:  # 100번째 배치마다 현재 손실과 학습 상태를 출력
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # 데이터셋의 전체 크기를 가져옴
    num_batches = len(dataloader)  # 데이터 로더의 배치 수를 가져옴
    model.eval()  # 모델을 평가 모드로 설정
    test_loss, correct = 0, 0  # 테스트 손실과 정확도 계산을 위한 변수 초기화

    with torch.no_grad():  # 기울기 계산을 비활성화(평가시 필요 없음)
        for X, y in dataloader:  # 데이터 로더에서 배치 단위로 데이터를 반복
            X, y = X.to(device), y.to(device)  # 입력 데이터(X)와 레이블(y)을 mps로 이동

            pred = model(X)  # 모델을 사용하여 예측값을 계산
            test_loss += loss_fn(pred, y).item()  # 배치 손실을 계산하고 누적
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # 예측값과 실제값을 비교하여 정확도를 계산하고 누적
        
        test_loss = test_loss/num_batches  # 평균 테스트 손실을 계산
        correct = correct/size  # 정확도를 계산

        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n") 


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-----------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print(f"Done!")


# 모델 저장하기
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# 모델 사용

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}, Actaul: {actual}"')