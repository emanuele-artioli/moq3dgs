import os
os.environ["NVCC_FLAGS"] = "-allow-unsupported-compiler"
import torch
import gsplat
means = torch.randn((10, 3), device="cuda:1")
quats = torch.randn((10, 4), device="cuda:1")
scales = torch.rand((10, 3), device="cuda:1") * 0.1
colors = torch.rand((10, 3), device="cuda:1")
opacities = torch.rand((10,), device="cuda:1")
viewmats = torch.eye(4, device="cuda:1")[None, :, :]
Ks = torch.tensor([[300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device="cuda:1")[None, :, :]
width, height = 300, 200
colors, alphas, meta = gsplat.rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height)
print("SUCCESS!")
