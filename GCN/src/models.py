import torch
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import torch_geometric

# ========================================
# torch / cuda 버전 
# ========================================
# conda install -c pytorch torchvision cudatoolkit=11.1 pytorch


# ========================================
# configuration.yaml
# ========================================
with open('configuration.yaml') as f:
  conf = yaml.load(f)

# ========================================
# 파일관련 
# ========================================

# ========================================
# 기타 utils
# ========================================

# ========================================
# main
# ========================================
def main():
  return 0


if __name__ == "__main__":
  main()