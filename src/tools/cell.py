"""Functions of cells"""
import mindspore.nn as nn
from mindspore import dtype as mstype

from src.args import args


def do_keep_fp32(network, cell_types):
    """Cast cell to fp32 if cell in cell_types"""
    for _, cell in network.cells_and_names():
        if isinstance(cell, cell_types):
            cell.to_float(mstype.float32)


def cast_amp(net):
    """cast network amp_level"""
    if args.amp_level == "O1":
        print(f"=> using amp_level {args.amp_level}\n"
              f"=> change {args.arch} to fp16")
        net.to_float(mstype.float16)
        cell_types = (nn.GELU, nn.Softmax, nn.Conv2d, nn.Conv1d, nn.BatchNorm2d, nn.LayerNorm)
        print(f"=> cast {cell_types} to fp32 back")
        do_keep_fp32(net, cell_types)
    elif args.amp_level == "O2":
        print(f"=> using amp_level {args.amp_level}\n"
              f"=> change {args.arch} to fp16")
        net.to_float(mstype.float16)
        cell_types = (nn.BatchNorm2d, nn.LayerNorm)
        print(f"=> cast {cell_types} to fp32 back")
        do_keep_fp32(net, cell_types)
    elif args.amp_level == "O3":
        print(f"=> using amp_level {args.amp_level}\n"
              f"=> change {args.arch} to fp16")
        net.to_float(mstype.float16)
    else:
        print(f"=> using amp_level {args.amp_level}")
        args.loss_scale = 1.
        args.is_dynamic_loss_scale = 0
        print(f"=> When amp_level is O0, using fixed loss_scale with {args.loss_scale}")
