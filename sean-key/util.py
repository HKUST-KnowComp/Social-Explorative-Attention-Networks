import argparse


def parse_args():

    parser = argparse.ArgumentParser(
        description="Run Social Exploration Attention Network.")
    parser.add_argument(
        '--explorative-method',
        nargs='?',
        default='RS-F1',
        help='RS-F1.')
    parser.add_argument(
        '--word-representation',
        nargs='?',
        default='../dataset/steemit/word2vec.txt',
        help='Word representation path')

    parser.add_argument(
        '--choice',
        nargs='?',
        default='cos',
        help='Similarity function choice. Default is cosine similarity.')

    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='validation split. Default is 0.1.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num-walks', type=int, default=3,
                        help='Number of walks per source. Default is 3.')

    parser.add_argument('--alpha', type=float, default=1,
                        help='Regularization hyperparameter. Default is 1.')

    parser.add_argument(
        '--output-dir',
        nargs='?',
        default='output/',
        help='Output directory.')

    parser.add_argument(
        '--embedding-matrix',
        nargs='?',
        default='embedding_matrix.npy',
        help='Pre-trained embedding matrix.')

    parser.add_argument(
        '--processed-feed',
        nargs='?',
        default='processed_feed',
        help='Load the preprocessed feed.')

    parser.add_argument('--max-user', type=int, default=7242,
                        help='Max number of users for steemit. Default is 7242.')

    parser.add_argument(
        '--day-count',
        type=int,
        default=370,
        help='Day count for steemit. Default is 370.')

    parser.add_argument(
        '--max-len-doc',
        type=int,
        default=90,
        help='Max number of keywords to represent an article. Default is 90.')

    parser.add_argument(
        '--max-len-user',
        type=int,
        default=200,
        help='Max number of keywords to represent a user. Default is 200.')

    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=300,
        help='Number of word embedding dimensions. Default is 300.')

    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Number of walks per source. Default is 64.')

    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout for model. Default is 0.2.')

    parser.add_argument('--epochs', type=int, default=3,
                        help='Epochs for training. Default is 3.')

    parser.add_argument('--batch-size', type=int, default=128,
                        help='Minibatch for training. Default is 128.')

    parser.add_argument('--use-social', type=int, default=1,
                        help='Add soical: 1, otherwise 0.')
    parser.add_argument('--use-sim', type=int, default=0,
                        help='Use Attention: 0, use similarity function 1.')
    print("type args", vars(parser.parse_args()))
    return parser.parse_args()

args = parse_args()
