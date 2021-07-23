from torch.utils.data import Dataset, DataLoader

# ========================================
# Custom Dataset
# ========================================
class WDDataset(Dataset):
    def __init__(self, X_wide, X_deep, Y):
        self.X_wide = X_wide
        self.X_deep = X_deep
        self.Y = Y
        
    def __len__(self):
        return len(self.X_wide)
    
    def __getitem__(self, index):
        return self.X_wide[index], self.X_deep[index], self.Y[index]