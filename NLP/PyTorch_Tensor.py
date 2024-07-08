import torch 
import numpy as np

# 데이터로 부터 직접 생성하기
data =[[1,2],[3,4]]
x_data = torch.tensor(data)

# Numpy 배열로부터 생성하기
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 다른 텐서로부터 생성하기
x_ones = torch.ones_like(x_data) # x_data의 속성을 유지한다
print(f"Ones Tensor: \n{x_ones}\n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어쓴다
print(f"Random Tensor: \n {x_rand} \n")

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor}\n")
print(f"Ones Tensor: \n {ones_tensor}\n")
print(f"Zeros Tensor: \n {zeros_tensor}\n")

# 텐서의 속성
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 텐서 연산

if torch.cuda.is_available():
    tensor = tensor.to("cuda")
# numpy식의 표준 인덱싱과 슬라이싱
tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
print(tensor)

# 텐서합치기
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 산술연산
# 두 텐서간의 행렬곱을 계산하자

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

#요소별 곱을 계산하자
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)


#단일 요소
agg = tensor.sum()
agg_item= agg.item()
print(agg_item,type(agg_item))

# 바꿔치기 연산
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# 텐서를 numpy배열로 변환하기
t= torch.ones(5)
print(f"t: {t}")
n= t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
