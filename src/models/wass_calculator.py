import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.functional as F
from geomloss import SamplesLoss

def compute_weighted_wasserstein_geomloss(A, B, wa, wb=None):
    """
    Parameters:
    - A: Tensor of shape (n, T, dim), samples from distribution A
    - B: Tensor of shape (n, T, dim), samples from distribution B
    - wa: Tensor of shape (n, T, 1), weights for each sample in A at each time
    - wb: Tensor of shape (n, T, 1), weights for each sample in B at each time, default is None
    """
    # print(A.shape, B.shape, wa.shape, wb.shape if wb is not None else None)
    if wb is None:
        wb = torch.ones_like(wa).to(wa.device)
        wb = F.softmax(wb, dim=0)
    wa = F.softmax(wa, dim=0)
    n, T, dim = A.shape
    distances = torch.zeros(T, 1)
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
    for t in range(T):
        distances[t] = loss(wa[:, t], A[:, t], wb[:, t], B[:, t])
    return distances.mean()

class WeightNetwork(nn.Module):
    def __init__(self, dim, hidden_dim=16):
        super(WeightNetwork, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)  
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

def train_model(A, B, epochs=5, batch_size=1024):
    n, T, dim = A.shape
    model = WeightNetwork(dim).cuda()
    optimizer = Adam(model.parameters(), lr=0.1)
    
    A, B = A.cuda(), B.cuda()
    dataset = TensorDataset(A, B)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            A_batch, B_batch = data
            optimizer.zero_grad()
            w = model(A_batch)
            # w = F.softmax(w, dim=0)
            # loss = weighted_mmd_distance(A_batch, B_batch, w)
            loss = compute_weighted_wasserstein_geomloss(A_batch, B_batch, w)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
    return model

def train_model_v2(A, B, epochs=10, batch_size=1024, device='cuda'):
    n, T, dim = A.shape
    model = WeightNetwork(dim)
    A, B = A.to(device), B.to(device)
    w = nn.Parameter(torch.randn(n, T, 1).to(A.device))
    optimizer = Adam([w], lr=0.1)
    for epoch in range(epochs):
        optimizer.zero_grad()

        loss = compute_weighted_wasserstein_geomloss(A, B, w)
        loss.backward()
        optimizer.step()
        
        rw = F.softmax(w, dim=0)
        

    return w


def get_weights(A, B, epoch=5):
    # w = train_model_v2(A, B)
    model = train_model(A, B, epochs=epoch)
    with torch.no_grad():
        w = model(A)

    return w

if __name__ == "__main__":
    n, T, dim = 3500, 60, 2 
    A = torch.randn(n, T, dim)
    B = torch.randn(n, T, dim)

    weights = get_weights(A, B)
    print("Computed Weights shape:", weights.shape)
