import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', type=str, default="../")

parser.add_argument('--train_batch_size', type=int, default=128)

parser.add_argument('--val_batch_size', type=int, default=64)

parser.add_argument('--test_batch_size', type=int, default=256)

parser.add_argument('--learning_rate', type=float, default=0.01)

parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--cuda', type=bool, default=True)

parser.add_argument('--create_graph_pairs', action="store_true")

parser.add_argument('--data_path', type=str, default="../data")