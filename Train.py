import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # ✅ 确保模型移动到 GPU
print(f"Using device: {device}")
