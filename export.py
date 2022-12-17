"""
##############export checkpoint file into air, onnx or mindir model#################
python export.py
"""

import numpy as np
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from mindspore import dtype as mstype

from src.args import args
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import get_model

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

if args.device_target in ["Ascend", "GPU"]:
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    net = get_model(args)
    criterion = get_criterion(args)
    cast_amp(net)
    net_with_loss = NetWithLoss(net, criterion)
    assert args.pretrained is not None, "checkpoint_path is None."

    param_dict = load_checkpoint(args.pretrained)
    load_param_into_net(net, param_dict)

    net.set_train(False)
    net.to_float(mstype.float32)

    input_arr = Tensor(np.zeros([1, 3, args.image_size, args.image_size], np.float32))
    export(net, input_arr, file_name=args.arch, file_format=args.file_format)
