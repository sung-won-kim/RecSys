import torch
from layers import VAE, loss_function
import yaml
import argparse
from torchvision.utils import save_image

# ========================================
# configuration.yaml
# ========================================
with open('configuration.yaml') as f:
  conf = yaml.load(f)

n_input = conf['n_input'] # 784
n_hidden1 = conf['n_hidden1'] # 512
n_hidden2 = conf['n_hidden2'] # 256
n_output =  conf['n_z'] # 2

# ========================================
# Generation settings
# ========================================
parser = argparse.ArgumentParser()
parser.add_argument('--num_predict', required = False, type=int, default=64,
                    help='# of prediction (default = 64)')
parser.add_argument('--fname', required = False, type=str, default='sample.png',
                    help='fname (default = sample.png)')

args = parser.parse_args()

# ========================================
# Load Model
# ========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE(n_input = n_input, n_hidden1 = n_hidden1, n_hidden2 = n_hidden2, n_output = n_output)

model.load_state_dict(torch.load('./model/vae.pt'))
model.to(device)


# ========================================
# Generate Samples
# ========================================
model.eval()

with torch.no_grad():
   # (0,1) Gaussian sampling  
   z = torch.randn(args.num_predict,n_output).to(device)
   sample = model.bernoulli_decoder(z)
   save_image(sample.view(args.num_predict,1,28,28), f'./sample/{args.fname}')

print('')
print(f'** GENERATION COMPLETE -- {args.fname} saved **')