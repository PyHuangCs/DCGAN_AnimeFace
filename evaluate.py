from train import losses_g, losses_d, real_scores, fake_scores
import matplotlib.pyplot as plt

# 生成器和判别器的损失
plt.figure(figsize=(15, 6))
plt.plot(losses_d, '-')
plt.plot(losses_g, '-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Discriminator', 'Generator'])
plt.title('Losses')
plt.savefig('evaluate_image/g-d_losses.png')

# 真实和生成样本的分数，real_scores趋于1&&fake_scores趋于0效果越好
plt.figure(figsize=(15, 6))
plt.plot(real_scores, '-')
plt.plot(fake_scores, '-')
plt.xlabel('epoch')
plt.ylabel('score')
plt.legend(['Real', 'Fake'])
plt.title('Scores')
plt.savefig('evaluate_image/real-fake_scores.png')