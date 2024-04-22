import torch

import utility
import data
import model
import loss
from option import args
# from trainer import Trainer
from trainer_wa_weight_fsrcnn_twostage import Trainer
import os

# torch.cuda.set_device(3)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    utility.print_network(model)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        # normal sampling
        t.train()
        # # inverse sampling
        # t.train_inverse()
        t.test()

    checkpoint.done()

