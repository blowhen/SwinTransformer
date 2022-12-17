"""LearningRate scheduler functions"""
import numpy as np

__all__ = ["multistep_lr", "cosine_lr", "constant_lr", "get_policy", "exp_lr"]


def get_policy(name):
    """get lr policy from name"""
    if name is None:
        return constant_lr

    out_dict = {
        "constant_lr": constant_lr,
        "cosine_lr": cosine_lr,
        "multistep_lr": multistep_lr,
        "exp_lr": exp_lr,
    }

    return out_dict[name]


def constant_lr(args, batch_num):
    """Get constant lr"""
    learning_rate = []

    def _lr_adjuster(epoch):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.warmup_lr, args.base_lr, args.warmup_length, epoch)
        else:
            lr = args.base_lr

        return lr

    for epoch in range(args.epochs):
        for batch in range(batch_num):
            learning_rate.append(_lr_adjuster(epoch + batch / batch_num))
    learning_rate = np.clip(learning_rate, args.min_lr, max(learning_rate))
    return learning_rate


def exp_lr(args, batch_num):
    """Get exp lr """
    learning_rate = []

    def _lr_adjuster(epoch):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.warmup_lr, args.base_lr, args.warmup_length, epoch)
        else:
            lr = args.base_lr * args.lr_gamma ** epoch

        return lr

    for epoch in range(args.epochs):
        for batch in range(batch_num):
            learning_rate.append(_lr_adjuster(epoch + batch / batch_num))
    learning_rate = np.clip(learning_rate, args.min_lr, max(learning_rate))
    return learning_rate


def cosine_lr(args, batch_num):
    """Get cosine lr"""
    learning_rate = []

    def _lr_adjuster(epoch):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.warmup_lr, args.base_lr, args.warmup_length, epoch)
        else:
            e = epoch - args.warmup_length
            es = args.epochs - args.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * args.base_lr

        return lr

    for epoch in range(args.epochs):
        for batch in range(batch_num):
            learning_rate.append(_lr_adjuster(epoch + batch / batch_num))
    learning_rate = np.clip(learning_rate, args.min_lr, max(learning_rate))
    return learning_rate


def multistep_lr(args, batch_num):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    learning_rate = []

    def _lr_adjuster(epoch):
        lr = args.base_lr * (args.lr_gamma ** (epoch / args.lr_adjust))
        return lr

    for epoch in range(args.epochs):
        for batch in range(batch_num):
            learning_rate.append(_lr_adjuster(epoch + batch / batch_num))
    learning_rate = np.clip(learning_rate, args.min_lr, max(learning_rate))
    return learning_rate


def _warmup_lr(warmup_lr, base_lr, warmup_length, epoch):
    """Linear warmup"""
    return epoch / warmup_length * (base_lr - warmup_lr) + warmup_lr
