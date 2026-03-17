import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')

    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--anchor-index", type=int, default=0)
    parser.add_argument("--data-folder", type=str, default="/home/ubuntu/projects/hjj-robot_lab/source/motion")
    parser.add_argument("--meta-folder", type=str, default="/home/ubuntu/projects/hjj-robot_lab/source/vq-vae")
    parser.add_argument("--out-dir", type=str, default="/home/ubuntu/projects/hjj-robot_lab/source/vq-vae/logs")
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume')
    parser.add_argument("--codebook", type=str, default=None)
    parser.add_argument("--data-process", type=int, default=False)

    # optim
    parser.add_argument('--total-iter', default=300000, type=int)
    parser.add_argument('--warm-up-iter', default=1000, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr-scheduler', default=[200000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")

    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss-vel', type=float, default=0.5, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons-loss', type=str, default='l1', help='reconstruction loss')

    # vq-vae
    parser.add_argument("--features", type=int, default=182)
    parser.add_argument("--code-dim", type=int, default=64, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=1024, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=1, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')

    return parser.parse_args()