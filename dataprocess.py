import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch
# 数据目录
DATA_DIR = '/root/autodl-tmp/DCGAN/data'

# 图像大小和批量大小
image_size = 64
batch_size = 128
#图像数据的均值和标准差
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# 数据预处理，调整图像大小、中心裁剪、转换为张量以及归一化
train_ds = ImageFolder(DATA_DIR, transform=tt.Compose([
    tt.Resize(image_size),
    tt.CenterCrop(image_size),
    tt.ToTensor(),
    tt.Normalize(*stats)]))
print(f"Number of images loaded: {len(train_ds)}")

# 数据加载器
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)

#反归一化
def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

def show_images(images, nmax=64):
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    import torch
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

# 获取设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 将数据加载器移动到设备
train_dl = DeviceDataLoader(train_dl, device)