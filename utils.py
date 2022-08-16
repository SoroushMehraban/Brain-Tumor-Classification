import json
import logging
import os

import torch

import config

logging.basicConfig(level=logging.INFO)


def load_checkpoint(checkpoint_file, model, optimizer, scheduler, lr):
    logging.info("Loading checkpoint...")
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint['scheduler'])

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        logging.info("Checkpoint loaded")
        return checkpoint['epoch']
    else:
        logging.info("Checkpoint not found")
        return 1


def save_checkpoint(model, optimizer, scheduler, next_epoch, filename):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": next_epoch
    }
    torch.save(checkpoint, filename)
    logging.info("Checkpoint saved")


def save_runtime_info(runtime_info):
    with open(config.RUNTIME_INFO_FILE, 'w') as fp:
        json.dump(runtime_info, fp)
