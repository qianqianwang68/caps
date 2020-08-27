import configargparse


def get_args():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--config', is_config_file=True, help='config file path')

    ## path options
    parser.add_argument('--datadir', type=str, help='the dataset directory')
    parser.add_argument("--logdir", type=str, default='./logs/', help='dir of tensorboard logs')
    parser.add_argument("--outdir", type=str, default='./out/', help='dir of output e.g., ckpts')
    parser.add_argument("--ckpt_path", type=str, default="",
                        help='specific checkpoint path to load the model from, '
                             'if not specified, automatically reload from most recent checkpoints')

    ## general options
    parser.add_argument("--exp_name", type=str, help='experiment name')
    parser.add_argument('--n_iters', type=int, default=200000, help='max number of training iterations')
    parser.add_argument('--phase', type=str, default='train', help='train/val/test')

    # data options
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--num_pts', type=int, default=500, help='num of points trained in each pair')
    parser.add_argument('--train_kp', type=str, default='mixed', help='sift/random/mixed')
    parser.add_argument('--prune_kp', type=int, default=1, help='if prune non-matchable keypoints')

    # training options
    parser.add_argument('--batch_size', type=int, default=6, help='input batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='base learning rate')
    parser.add_argument("--lrate_decay_steps", type=int, default=80000,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--lrate_decay_factor", type=float, default=0.5,
                        help='decay learning rate by a factor every specified number of steps')

    ## model options
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone for feature representation extraction. supported: resent')
    parser.add_argument('--pretrained', type=int, default=1,
                        help='if use ImageNet pretrained weights to initialize the network')
    parser.add_argument('--coarse_feat_dim', type=int, default=128,
                        help='the feature dimension for coarse level features')
    parser.add_argument('--fine_feat_dim', type=int, default=128,
                        help='the feature dimension for fine level features')
    parser.add_argument('--prob_from', type=str, default='correlation',
                        help='compute prob by softmax(correlation score), or softmax(-distance),'
                             'options: correlation|distance')
    parser.add_argument('--window_size', type=float, default=0.125,
                        help='the size of the window, w.r.t image width at the fine level')
    parser.add_argument('--use_nn', type=int, default=1, help='if use nearest neighbor in the coarse level')

    ## loss function options
    parser.add_argument('--std', type=int, default=1, help='reweight loss using the standard deviation')
    parser.add_argument('--w_epipolar_coarse', type=float, default=1, help='coarse level epipolar loss weight')
    parser.add_argument('--w_epipolar_fine', type=float, default=1, help='fine level epipolar loss weight')
    parser.add_argument('--w_cycle_coarse', type=float, default=0.1, help='coarse level cycle consistency loss weight')
    parser.add_argument('--w_cycle_fine', type=float, default=0.1, help='fine level cycle consistency loss weight')
    parser.add_argument('--w_std', type=float, default=0, help='the weight for the loss on std')
    parser.add_argument('--th_cycle', type=float, default=0.025,
                        help='if the distance (normalized scale) from the prediction to epipolar line > this th, '
                             'do not add the cycle consistency loss')
    parser.add_argument('--th_epipolar', type=float, default=0.5,
                        help='if the distance (normalized scale) from the prediction to epipolar line > this th, '
                             'do not add the epipolar loss')

    ## logging options
    parser.add_argument('--log_scalar_interval', type=int, default=20, help='print interval')
    parser.add_argument('--log_img_interval', type=int, default=500, help='log image interval')
    parser.add_argument("--save_interval", type=int, default=10000, help='frequency of weight ckpt saving')

    ## eval options
    parser.add_argument('--extract_img_dir', type=str, help='the directory of images to extract features')
    parser.add_argument('--extract_out_dir', type=str, help='the directory of images to extract features')

    args = parser.parse_known_args()[0]

    return args
