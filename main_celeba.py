import argparse
from pathlib import Path
from train.celeba_train import run_train



def main():
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=3, help='img channel')
    parser.add_argument('--z_dim', type=int, default=32, help='the size of the img')
    parser.add_argument('--dataset', type=str, default='CelebA', help='the name of data')
    parser.add_argument('--epoch', type=int, default=60, help='training epoch')
    parser.add_argument('--save_idx', type=int, default=30, help='save')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=int, default=1e-4, help='batch size')
    parser.add_argument('--beta', type=float, default=10., help='beta')
    parser.add_argument('--recon', type=str, default='gaussian', help='recon')
    args = parser.parse_args()
    run_train(args)


if __name__ == '__main__':
    main()
