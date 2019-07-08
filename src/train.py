import os
import shutil
import datetime
import argparse

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
import numpy as np

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from tqdm import tqdm

from dataset import LMDBDataset

import logging
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=25)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lmdb-file', type=str, default='lmdb_train')
    parser.add_argument('--log-dir', type=str, default='logs')
    return parser.parse_args()


def copy_to_tmp(root):
    tmp_root = os.path.join('tmp', os.path.basename(root))
    if not os.path.exists(tmp_root):
        print('Copying {} to {}...'.format(root, tmp_root))
        shutil.copytree(root, tmp_root)
        print('Done.')
    return tmp_root


def get_data_loaders(root, input_size, batch_size, num_workers, device):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    root = copy_to_tmp(root)
    dataset = LMDBDataset(root=root, transform=transform)
    dataset_size = len(dataset)

    indices = list(range(dataset_size))
    split = int(np.floor(dataset_size * 0.8))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[:split], indices[split:]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_indices)

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if device.type == 'cuda' else {}
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, **kwargs)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, **kwargs)

    return train_loader, valid_loader


def build_model(num_classes):
    model = models.resnet101(pretrained=True)

    # for param in model.parameters():
    #     param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, valid_loader = get_data_loaders(args.lmdb_file, args.input_size, args.batch_size, args.num_workers, device)
    model = build_model(num_classes=len(train_loader.dataset.classes))
    print(model)

    log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device, non_blocking=True)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'loss': Loss(criterion)}, device=device)

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % 10 == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(10)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        # evaluator.run(train_loader)
        # metrics = evaluator.state.metrics
        # avg_accuracy = metrics['accuracy']
        # avg_loss = metrics['loss']
        # tqdm.write("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        #            .format(engine.state.epoch, avg_accuracy, avg_loss))

        writer.add_scalar('train_loss', engine.state.output, engine.state.epoch)
        writer.file_writer.flush()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        tqdm.write("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                   .format(engine.state.epoch, avg_accuracy, avg_loss))

        pbar.n = pbar.last_print_n = 0

        writer.add_scalar('val_loss', avg_loss, engine.state.epoch)
        writer.add_scalar('val_acc', avg_accuracy, engine.state.epoch)
        writer.file_writer.flush()

    trainer.run(train_loader, max_epochs=args.num_epochs)

    pbar.close()
    writer.close()


if __name__ == '__main__':
    main()
