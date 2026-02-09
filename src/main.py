import argparse
from har_classifier import main as run_main

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HAR Classifier Training Script')
    parser.add_argument('--dataset', type=str, default='data_5', help='Dataset name (e.g., data_5)')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate (e.g., 0.5)')
    parser.add_argument('--k_folds', type=int, default=1, help='Number of k-folds (e.g., 1)')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate (e.g., 1e-5)')
    parser.add_argument('--modeltype', type=str, default='har_classifier', help='Type of model (e.g., har_classifier)')
    parser.add_argument('--n_batch', type=int, default=256, help='Batch size (e.g., 256)')
    parser.add_argument('--n_epochs', type=int, default=3, help='Number of epochs (e.g., 100)')
    parser.add_argument('--norm_method', type=int, default=0, help='Normalization method (e.g., 0)')
    parser.add_argument('--normalize', type=str2bool, default=True, help='Normalize flag (True/False, e.g., True)')
    parser.add_argument('--overlap', type=str2bool, default=True, help='Overlap flag (True/False, e.g., True)')
    parser.add_argument('--overlap_shift', type=float, default=0.5, help='Overlap shift (e.g., 0.5)')
    parser.add_argument('--seg5', type=str2bool, default=True, help='Segment 5 flag (True/False, e.g., True)')
    parser.add_argument('--sensors', type=int, default=3, help='Number of sensors (e.g., 3)')
    parser.add_argument('--test_type', type=str, default='nusers', help='Type of test (e.g., nusers)')
    parser.add_argument('--with_val', type=str2bool, default=True, help='With validation flag (True/False, e.g., True)')

    args = parser.parse_args()

    run_main(
        args.modeltype,
        args.test_type,
        args.k_folds,
        args.norm_method,
        args.n_epochs,
        args.dataset,
        args.learning_rate,
        args.dropout_rate,
        args.overlap_shift,
        args.n_batch,
        args.sensors,
        args.seg5,
        args.overlap,
        args.normalize,
        args.with_val
    )