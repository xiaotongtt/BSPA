import argparse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=16,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

'''
action='store_true'
Note：有default值的时候，running时不声明就为默认值，

没有的话，如果是store_false,则默认值是True，如果是store_true,则默认值是False
'''

# Data specifications
# parser.add_argument('--dir_data', type=str, default='/home/yulun/data/SR/traindata/DIV2K/bicubic',
#                     help='dataset directory')

parser.add_argument('--dir_data', type=str, default='/data/xxx',
                    help='dataset directory')
# parser.add_argument('--dir_data', type=str, default='/media/lxt/data/SR/RCAN-master/dataset',
#                     help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
# normal and inverse sampling
parser.add_argument('--data_train_normal', type=str, default='DIV2K_NORMAL',
                    help='train dataset name')
parser.add_argument('--data_train_inverse', type=str, default='DIV2K_INVERSE',
                    help='train dataset name')

parser.add_argument('--data_test', type=str, default='Set14',  # 'Set14'    DIV2K_valid
                    help='test dataset name')
parser.add_argument('--benchmark_noise', action='store_true',
                    help='use noisy benchmark sets')
parser.add_argument('--n_train', type=int, default=800,
                    help='number of training set')
# parser.add_argument('--n_val', type=int, default=5,
#                     help='number of validation set')
parser.add_argument('--n_val', type=int, default=14,
                    help='number of validation set')
parser.add_argument('--offset_val', type=int, default=800,
                    help='validation index offest')
parser.add_argument('--ext', type=str, default='img',  # sep
                    help='dataset file extension')
parser.add_argument('--scale', default='4',  # 2,3,4,8
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=128,  #192,  256
                    help='output patch size')
# parser.add_argument('--patch_size', type=int, default=96,
#                     help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--noise', type=str, default='.',
                    help='Gaussian noise std.')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')

# Model specifications
parser.add_argument('--model', default='EDSR_BASELINE',  # EDSR_BASELINE
                    help='model name')
parser.add_argument('--un_factor', type=str, default='10',  #192,  256 100
                    help='unbalanced factor')
# parser.add_argument('--model', default='EDSR_BASELINE',
#                     help='model name')  # RCAN_LATTICE_BASE_FUSION_COVA
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',  #
                    help='pre-trained model directory')
# parser.add_argument('--pre_train', type=str, default='experiment/model/RCAN_BIX4.pt',
#                     help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Training specifications
# parser.add_argument('--reset', action='store_true',
#                     help='reset the training')
parser.add_argument('--reset', default=False, action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=400,  # 1000 400
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,  # 16
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', default=False, action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=2e-4, #2e-4, 1e-4, 2.5e-5
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',  # step cosine
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Loss specifications
parser.add_argument('--alpha', type=float, default=0.5,
                    help='sin')
parser.add_argument('--loss', type=str, default='1*L1',  ### 0.005*DCT
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# Log specifications
# parser.add_argument('--save', type=str, default='test',
#                     help='file name to save')
parser.add_argument('--save', type=str, default='../EDSR/AB_weight_un10_0.1_inter10_Sy1',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
# parser.add_argument('--print_model', action='store_true',
#                     help='print model')
parser.add_argument('--print_model', action='store_true',
                    help='print model')
parser.add_argument('--save_models', default=False, action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=10,  # 100
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true', default=True,
                    help='save output results')

# options for residual group and feature channel reduction
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
# options for test
parser.add_argument('--testpath', type=str, default='../test/DIV2K_val_LR_our',
                    help='dataset directory for testing')
parser.add_argument('--testset', type=str, default='Set5',
                    help='dataset name for testing')

# continual learning
parser.add_argument('--task_id', type=int, default=0,
                    help='task number')
parser.add_argument('--save_path', type=str, default='/data/xxx/BSPA/experiment/EDSR/',
                    help='dataset directory for saving the importance')

parser.add_argument('--ssim', action='store_true', default=False,
                    help='caculate and log SSIM (time-consuming for large image)')
parser.add_argument("--step", type=int, default=112,  # 28x4 180
                    help='split-merge step')
parser.add_argument("--n_parallel", type=int, default=500,
                    help='number of patches when using APE to restore LR patches in parallel')
parser.add_argument('--save_gt', action='store_true', default=False,
                    help='save low-resolution and high-resolution images together')

parser.add_argument('--remain_factor', type=float, default=0.1,  # 0.95 0.01
                    help='the remain weigt parameters')
parser.add_argument('--interval', type=int, default=50,  # 0.95 0.01
                    help='the remain weigt parameters')
parser.add_argument('--dilation', type=int, default=None,  # 0.95 0.01
                    help='the remain weigt parameters')
parser.add_argument("--number", type=int, default=2500,  # 500
                    help='the numner of inverse data')
                    
parser.add_argument('--chunk_size', type=int, default=144,  # 0.95 0.01
                    help='the remain weigt parameters')
parser.add_argument('--n_hashes', type=int, default=4,  # 0.95 0.01
                    help='the remain weigt parameters')

# add for IG
parser.add_argument(
    '--baseline_model_path',
    type=str,
    default='/home/vip/DATA4/xxx/Longtail_new/experiment/EDSR/normal/edsr_normal_ps32/model/model_latest.pt',
    help='path of baseline model')
parser.add_argument(
    '--target_model_path', type=str,
    default='/home/vip/DATA4/xxx/Longtail_new/experiment/EDSR/pretrain_model/edsr_inverse_un100_ps32_pretrain_normal/model/model_best.pt',
    help='path of target model')
parser.add_argument('--gt_folder', type=str, default='/home/vip/DATA4/xxx/dataset/DIV2K/DIV2K_train_HR_crop/X4/',
                    help='folder that contains gt image')  # datasets/Set14/GTmod12
parser.add_argument(
    '--blur_folder', type=str, default='/data/xxx/DIV2K/DIV2K_train_LR_bicubic_crop/X4/',
    help='folder that contains blurry image')  # datasets/Set14/Blur2_LRbicx2
parser.add_argument(
    '--noise_folder', type=str, default='/data/xxx/DIV2K/DIV2K_train_LR_bicubic_crop/X4/',
    help='folder that contains noisy image')  # datasets/Set14/LRbicx2_noise0.1
parser.add_argument('--total_step', type=int, default=100)
# parser.add_argument('--scale', type=int, default=4, help='scale ratio')  # 2
parser.add_argument(
    '--record_filters_folder',
    type=str,
    default='results/edsr2/faig',  # results/Interpret/neuron-search/srresnet/Set14/faig
    help='folder that saves the sorted location index of discovered filters')

args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

