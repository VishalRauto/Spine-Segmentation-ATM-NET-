import torch
print('PyTorch   :', torch.__version__)
print('CUDA      :', torch.version.cuda)
print('GPU avail :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU name  :', torch.cuda.get_device_name(0))
    print('VRAM      :', torch.cuda.get_device_properties(0).total_memory // 1024**2, 'MB')
    x = torch.randn(512,512).cuda()
    print('GPU test  : OK -', x.device)
else:
    print('CUDA NOT available - CPU only mode')
