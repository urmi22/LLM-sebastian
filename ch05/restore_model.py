
import torch
from ch04.GPT_model import GPTModel
from ch05.GPT_CONFIG_124M import GPT_2_small as cfg


# Restore the model and optimizer states by first loading the saved data via  torch.load and then using the load_state_dict method
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("./ch05/model_and_optimizer.pth", map_location=device)
model = GPTModel(cfg)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()