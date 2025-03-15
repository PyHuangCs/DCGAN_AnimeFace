import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dataprocess import train_dl, denorm,device,batch_size
from model import discriminator, generator, latent_size
from dataprocess import to_device
import seaborn as sns
sns.set(style='darkgrid')

# 生成样本保存目录
sample_dir = 'generated_images'
os.makedirs(sample_dir, exist_ok=True)

discriminator = to_device(discriminator, device)
generator = to_device(generator, device)

# 固定的潜在向量
latent_size = 128
fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)

# 保存生成的样本
def save_samples(total_epochs, index, latent_tensors, show=True,sample_dir='generated_images'):
    fake_images = model["generator"](latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}-{1:0=4d}.png'.format(total_epochs, index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

# 训练模型
def fit(model, criterion, epochs, lr, start_idx=1):
    model["discriminator"].train()
    model["generator"].train()
    torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Create optimizers
    optimizer = {
        "discriminator": torch.optim.Adam(model["discriminator"].parameters(), lr=lr, betas=(0.5, 0.999)),
        "generator": torch.optim.Adam(model["generator"].parameters(),lr=lr, betas=(0.5, 0.999))
    }

    # Learning rate schedulers
    #scheduler_d = StepLR(optimizer["discriminator"], step_size=30, gamma=0.1)
    #scheduler_g = StepLR(optimizer["generator"], step_size=30, gamma=0.8)

    for epoch in range(epochs):
        loss_d_per_epoch = []
        loss_g_per_epoch = []
        real_score_per_epoch = []
        fake_score_per_epoch = []
        for real_images, _ in tqdm(train_dl):
            # Train discriminator
            # Clear discriminator gradients
            optimizer["discriminator"].zero_grad()

            # Pass real images through discriminator
            real_preds = model["discriminator"](real_images)
            real_targets = torch.ones(real_images.size(0), 1, device=device)
            real_loss = criterion["discriminator"](real_preds, real_targets)
            cur_real_score = torch.mean(real_preds).item()
            
            # Generate fake images
            latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
            fake_images = model["generator"](latent)

            # Pass fake images through discriminator
            fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
            fake_preds = model["discriminator"](fake_images)
            fake_loss = criterion["discriminator"](fake_preds, fake_targets)
            cur_fake_score = torch.mean(fake_preds).item()

            real_score_per_epoch.append(cur_real_score)
            fake_score_per_epoch.append(cur_fake_score)

            # Update discriminator weights
            loss_d = real_loss + fake_loss
            loss_d.backward()
            optimizer["discriminator"].step()
            loss_d_per_epoch.append(loss_d.item())


            # Train generator
            # Clear generator gradients
            optimizer["generator"].zero_grad()
            
            # Generate fake images
            latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
            fake_images = model["generator"](latent)
            
            # Try to fool the discriminator
            preds = model["discriminator"](fake_images)
            targets = torch.ones(batch_size, 1, device=device)
            loss_g = criterion["generator"](preds, targets)
            
            # Update generator weights
            loss_g.backward()
            optimizer["generator"].step()
            loss_g_per_epoch.append(loss_g.item())
            
        # Record losses & scores
        losses_g.append(np.mean(loss_g_per_epoch))
        losses_d.append(np.mean(loss_d_per_epoch))
        real_scores.append(np.mean(real_score_per_epoch))
        fake_scores.append(np.mean(fake_score_per_epoch))
        
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, 
            losses_g[-1], losses_d[-1], real_scores[-1], fake_scores[-1]))

        # Adjust learning rates
        #scheduler_d.step()
        #scheduler_g.step()

        # Save generated images every epoch
        save_samples(epochs, epoch + start_idx, fixed_latent, show=False)
    
    return losses_g, losses_d, real_scores, fake_scores

# 初始化模型和损失函数
model = {
    "discriminator": discriminator.to(device),
    "generator": generator.to(device)
}

criterion = {
    "discriminator": nn.BCELoss(),
    "generator": nn.BCELoss()
}

# 训练参数
lr = 0.0002
epochs = 60

# 开始训练
history = fit(model, criterion, epochs, lr)
losses_g, losses_d, real_scores, fake_scores = history
# 保存训练好的模型
torch.save(model["discriminator"].state_dict(), 'weight/discriminator.pth')
torch.save(model["generator"].state_dict(), 'weight/generator.pth')