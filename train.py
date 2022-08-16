import json
import logging
import os
from glob import glob

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BrainTumorDataset
import numpy as np
import torch.optim
import config
from utils import load_checkpoint, save_checkpoint, save_runtime_info

logging.basicConfig(level=logging.INFO)


def step_scheduler(epoch):
    if epoch < config.EPOCHS * config.SCHEDULER_STEP1:
        factor = 1
    elif config.EPOCHS * config.SCHEDULER_STEP1 <= epoch < config.EPOCHS * config.SCHEDULER_STEP2:
        factor = 0.1
    else:
        factor = 0.01
    return factor


def get_loaders():
    np.random.seed(config.RANDOM_SEED)
    images = np.array(glob('final_dataset/*'))
    np.random.shuffle(images)

    images_len = images.shape[0]
    train_images, validation_images, test_images = np.split(images,
                                                            [int(config.TRAIN_RATIO * images_len),
                                                             int((config.TRAIN_RATIO + config.VALIDATION_RATIO)
                                                                 * images_len)])

    train_dataset = BrainTumorDataset(train_images.tolist(), transform=config.transform)
    validation_dataset = BrainTumorDataset(validation_images.tolist(), transform=config.transform)
    test_dataset = BrainTumorDataset(test_images.tolist(), transform=config.transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=config.BATCH_SIZE,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )

    return train_loader, validation_loader, test_loader


def train_fn(train_loader, model, optimizer, loss_fn):
    losses = []

    for x, y in train_loader:
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_loss = sum(losses) / len(losses)
    return mean_loss


@torch.no_grad()
def validation_fn(validation_loader, model, loss_fn):
    model.eval()

    losses = []
    for x, y in validation_loader:
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        losses.append(loss.item())

    model.train()

    mean_loss = sum(losses) / len(losses)
    return mean_loss


@torch.no_grad()
def accuracy_fn(loader, model):
    model.eval()

    correct_preds, total = 0, 0
    for x, y in loader:
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        out = model(x)

        out_classes = torch.max(out, dim=1)[1]

        is_correct = out_classes == y
        correct_preds += torch.sum(is_correct).item()
        total += is_correct.shape[0]

    model.train()

    return correct_preds / total


def main():
    if os.path.exists(config.RUNTIME_INFO_FILE):
        with open(config.RUNTIME_INFO_FILE, 'r') as fp:
            runtime_info = json.load(fp)
    else:
        runtime_info = {
            "batch_size": config.BATCH_SIZE,
            "initial_lr": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            'train_losses': [],
            'validation_losses': [],
            'accuracies': [],
        }

    model = config.model
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, step_scheduler)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader, validation_loader, test_loader = get_loaders()

    if config.LOAD_MODEL:
        current_epoch = load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, scheduler,
                                        config.LEARNING_RATE)
    else:
        current_epoch = 1

    loop = tqdm(range(current_epoch, config.EPOCHS + 1), leave=True)
    for epoch in loop:
        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        scheduler.step()

        current_lr = scheduler.get_last_lr()
        loop.set_postfix(loss=train_loss, lr=current_lr)
        loop.set_description(f"Epoch [{epoch}]")

        if epoch % 10 == 0 or epoch == 1:
            if config.SAVE_MODEL:
                save_checkpoint(model, optimizer, scheduler, current_epoch + 1, filename=config.CHECKPOINT_FILE)

            validation_loss = validation_fn(validation_loader, model, loss_fn)
            accuracy = accuracy_fn(validation_loader, model)

            runtime_info['train_losses'].append((epoch, train_loss))
            runtime_info['validation_losses'].append((epoch, validation_loss))
            runtime_info['accuracies'].append((epoch, accuracy))
            save_runtime_info(runtime_info)

            logging.info(f"Train loss: {train_loss} | Validation loss: {validation_loss}"
                         f" | accuracy: {accuracy * 100:.2f}%")

    save_checkpoint(model, optimizer, scheduler, config.EPOCHS + 1, filename=config.CHECKPOINT_FILE)
    logging.info("Final Evaluation")
    accuracy = accuracy_fn(test_loader, model)
    logging.info(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()
