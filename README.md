# DCGAN Anime Face Generation

A PyTorch implementation of DCGAN for generating 64x64 anime face images.

## Features

- Deep Convolutional GAN (DCGAN)
- Anime face image generation
- PyTorch-based training pipeline
- GPU acceleration support
- Generated sample visualization
- Model checkpoint saving

## Project Structure

```text
DCGAN_AnimeFace/
├── dataprocess.py      # Dataset loading and preprocessing
├── model.py            # Generator and discriminator definitions
├── train.py            # Training script
├── generated_images/   # Generated samples during training
├── weight/             # Saved model checkpoints
└── data/               # Training dataset (not recommended to commit)
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run training:

```bash
python train.py
```

## Model Architecture

### Generator

The generator transforms a 128-dimensional latent vector into a 64x64 RGB anime face image using transposed convolutions.

### Discriminator

The discriminator classifies whether an input anime face image is real or generated.

## Improvements Included

- Refactored model definitions into reusable classes
- Added weight initialization utility
- Improved project documentation
- Better repository structure recommendations
- Cleaner engineering practices

## Future Improvements

- Add configuration file support
- Add FID / IS evaluation metrics
- Add WGAN-GP support
- Add TensorBoard logging
- Add checkpoint resume support
- Add inference script for image generation

## Notes

Large datasets and generated outputs should not be committed to GitHub. Use `.gitignore` to exclude:

- `data/`
- `generated_images/`
- `weight/`
- `__pycache__/`

## License

MIT License
