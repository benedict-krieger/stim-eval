import argparse
import surprisal as surp



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--user',help=f'User name, for which a sub-directory must exist')
    parser.add_argument('--exp',help = f'Name of experiment and input tsv file')
    args = parser.parse_args()

    surp.merge_surprisal(args.user, args.exp)