import io
import torch.utils.data
import numpy as np
from PIL import Image
import lmdb


class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(LMDBDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform

        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.n = txn.stat()['entries'] // 2
            self.classes = np.unique([txn.get('label-{}'.format(idx).encode()).decode() for idx in range(self.n)])

        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        print('Found {} images belonging to {} classes'.format(self.n, len(self.classes)))

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            data = txn.get('image-{}'.format(index).encode())
            label = txn.get('label-{}'.format(index).encode()).decode()

        sample = Image.open(io.BytesIO(data))
        if self.transform is not None:
            sample = self.transform(sample)

        target = self.class_to_idx[label]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return self.n
