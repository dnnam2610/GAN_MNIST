# GAN MNIST Project

This repository contains code for training a GAN on the MNIST dataset.

## Links to Resources

- **Checkpoints:** [Google Drive folder](https://drive.google.com/drive/folders/1WTKCBZQzR2tjcwAe5fj-kxjklXO8QbyD?usp=sharing)  
  Contains saved models (`best.pt`, `last.pt`) from training.

- **Training Logs (TensorBoard):** [Google Drive folder](https://drive.google.com/drive/folders/1cRaWUNuZc_WNUFM03OFMxi4XoRNQfwj_?usp=sharing)  
  TensorBoard event files for monitoring training progress.

- **Generated Images:** [Google Drive folder](https://drive.google.com/drive/folders/1A1vWa5t-xRO34sg3_JhoOWbdW1MjRbHF?usp=sharing)  
  Sample images generated after training.

## Usage

1. Clone the repository.
2. Download the checkpoints and place them in the `checkpoints/` folder.
3. Run the inference script to generate images:

```bash
python inference.py
