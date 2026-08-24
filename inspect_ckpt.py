import torch
ckpt = torch.load('checkpoints/factorial_1000000_s42.pt', map_location='cpu')
print("Keys:", ckpt.keys())
model_state = ckpt.get("model_state_dict", ckpt)
for k, v in list(model_state.items())[:3]:
    print(k, v.shape, v.dtype)
