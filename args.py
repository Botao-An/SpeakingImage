from argparse import ArgumentParser

def get_arguments():

    parser = ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=['train','test','example'],
        default='test',
        help=("train: performs training and validation; test: tests the model"
              "Default: train ")
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="The batch size. Default: 4"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of the training epochs. Default: 10"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="The learning rate. Default: 1e-4"
    )

    parser.add_argument(
        "--lr-decay-epochs",
        type=int,
        default=5,
        help="The number of epochs before adjusting the learning rate. Default: 3"
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
        help="L2 regularization factor. Default: 1e-3"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="/extdisk",
        help="The root path of the data. Default: /extdisk/CoverData"
    )

    parser.add_argument(
        "--device",
        default='cuda',
        help="Device on which the network will be trained. Default: cuda"
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default='save/folder',
        help="The directory where models are saved. Default: save"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="The image height. Default: 256"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="The image width. Default: 256"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. Default: 4"
    )

    parser.add_argument(
        "--name",
        type=str,
        default='DaZhuangNet',
        help="Name given to the model when saving. Default: DaZhuangNet"
    )

    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.1,
        help="The learning rate decay factor. Default: 0.1"
    )

    return parser.parse_args()