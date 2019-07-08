import io
import argparse
from pathlib import Path
from multiprocessing import Process, Queue

import numpy as np
from PIL import Image
import lmdb
from tqdm import trange


MAX_SIZE = int(1e12)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset-dir', type=str, default='data/train')
parser.add_argument('--lmdb-file', type=str, default='lmdb_train')
parser.add_argument('--num-workers', type=int, default=8)
args = parser.parse_args()

p = Path(args.dataset_dir)
images = list((p.glob('*/*.jpg')))
num_images = len(images)
b = int(np.ceil(num_images / args.num_workers))


def worker(images, q):
    for impath in images:
        image = Image.open(impath)
        image = image.resize((256, 256), resample=Image.BICUBIC)
        data = io.BytesIO()
        image.save(data, format='JPEG', quality=95)
        label = impath.parts[-2]
        q.put((data, label))


q = Queue()
jobs = []
for i in range(args.num_workers):
    p = Process(target=worker, args=(images[b*i:b*(i+1)], q))
    p.start()
    jobs.append(p)

env = lmdb.open(args.lmdb_file, map_size=MAX_SIZE)
with env.begin(write=True) as txn:
    for idx in trange(num_images):
        data, label = q.get()
        image_key = 'image-{}'.format(idx)
        label_key = 'label-{}'.format(idx)
        txn.put(image_key.encode(), data.getvalue())
        txn.put(label_key.encode(), label.encode())

for p in jobs:
    p.join()

print('Done.')
