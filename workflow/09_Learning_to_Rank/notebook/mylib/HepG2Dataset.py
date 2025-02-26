from torch.utils.data import Dataset
import random
import numpy as np

class HepG2Dataset(Dataset):
    def __init__(self, total_samples, features, indexes):
        self.total_samples = total_samples
        self.features = features
        self.indexes = indexes

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        r1 = random.choice(self.indexes)
        r2 = random.choice(self.indexes)
        
        feature_1 = self.features[r1, 2]
        feature_2 = self.features[r2, 2]
        feature = np.concatenate((feature_1, feature_2), axis=0)
        
        val_1 = self.features[r1, 3]
        val_2 = self.features[r2, 3]
        
        y = 1 if val_1 > val_2 else 0
        
        return feature, y

class HepG2DatasetMedian(Dataset):
    def __init__(self, total_samples, features, indexes, median_features):
        self.total_samples = total_samples
        self.features = features
        self.indexes = indexes
        self.median_features = median_features

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        order = random.randint(0, 1)

        r = random.choice(self.indexes)
        feature_rand = self.features[r, 2]

        if (order == 0):
            feature_1 = self.median_features[0, 2]
            feature_2 = feature_rand
            val_1 = self.median_features[0, 3]
            val_2 = self.features[r, 3]
        else:
            feature_1 = feature_rand
            feature_2 = self.median_features[0, 2]
            val_1 = self.features[r, 3]
            val_2 = self.median_features[0, 3]

        feature = np.concatenate((feature_1, feature_2), axis=0)

        y = 1 if val_1 > val_2 else 0
        
        return feature, y

class HepG2DatasetTransformer(Dataset):
    def __init__(self, total_samples, features, indexes):
        self.total_samples = total_samples
        self.features = features
        self.indexes = indexes

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        r1 = random.choice(self.indexes)
        r2 = random.choice(self.indexes)
        
        feature_1 = self.features[r1, 4]
        feature_2 = self.features[r2, 4]
        feature = np.concatenate((feature_1, feature_2), axis=0)
        
        val_1 = self.features[r1, 3]
        val_2 = self.features[r2, 3]
        
        y = 1 if val_1 > val_2 else 0
        
        return feature, y

class HepG2DatasetAutoEncoder(Dataset):
    def __init__(self, total_samples, features, indexes):
        self.total_samples = total_samples
        self.features = features
        self.indexes = indexes

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        r1 = random.choice(self.indexes)
        r2 = random.choice(self.indexes)
        
        feature_1 = self.features[r1, 4]
        feature_2 = self.features[r2, 4]
        feature = np.concatenate((feature_1, feature_2), axis=0)
        
        val_1 = self.features[r1, 3]
        val_2 = self.features[r2, 3]
        
        y = 1 if val_1 > val_2 else 0
        
        return feature, y