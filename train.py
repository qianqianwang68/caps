import os
import config
from tensorboardX import SummaryWriter
from CAPS.caps_model import CAPSModel
from dataloader.megadepth import MegaDepthLoader
from utils import cycle


def train_megadepth(args):
    # save a copy for the current args in out_folder
    out_folder = os.path.join(args.outdir, args.exp_name)
    os.makedirs(out_folder, exist_ok=True)
    f = os.path.join(out_folder, 'args.txt')
    with open(f, 'w') as file:
        for arg in vars(args):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    # tensorboard writer
    tb_log_dir = os.path.join(args.logdir, args.exp_name)
    print('tensorboard log files are stored in {}'.format(tb_log_dir))
    writer = SummaryWriter(tb_log_dir)

    # megadepth data loader
    train_loader = MegaDepthLoader(args).load_data()
    train_loader_iterator = iter(cycle(train_loader))

    # define model
    model = CAPSModel(args)
    start_step = model.start_step

    # training loop
    for step in range(start_step + 1, start_step + args.n_iters + 1):
        data = next(train_loader_iterator)
        model.set_input(data)
        model.optimize_parameters()
        model.write_summary(writer, step)
        if step % args.save_interval == 0 and step > 0:
            model.save_model(step)


if __name__ == '__main__':
    args = config.get_args()
    train_megadepth(args)




