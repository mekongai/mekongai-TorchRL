import torch
from torch import nn, optim


def initialize_model():
    model = nn.Sequential(nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, 9))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


def train_model(model, optimizer, memory, reward):
    for state, action in memory:
        # Chuyển đổi trạng thái và hành động thành Tensor
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32)
        action_tensor = torch.tensor([action], dtype=torch.long)

        # Tính toán loss và tối ưu hóa mô hình
        q_values = model(state_tensor)
        target = q_values.clone()
        target[action_tensor] = reward
        loss = nn.functional.mse_loss(q_values, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
