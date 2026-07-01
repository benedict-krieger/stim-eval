import fasttext.util
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', required=True)
    args = parser.parse_args()

    fasttext.util.download_model(f'{args.lang}', if_exists='ignore')

