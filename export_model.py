import argparse
import os
from os import path as osp

import paddle
from paddle import inference
from paddle.inference import Config, create_predictor
from paddle.jit import to_static
from paddle.static import InputSpec
from paddle.vision import transforms
from paddlevideo.utils import get_config
from PIL import Image
from utils import build_inference_helper

from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('-c',
                    '--checkpoint',
                    type=str,
                    required=True,
                    help='path to checkpoint, e.g. ./logs/model-100.pdparams')


def _trans(path_to_checkpoint_file):
    model = Model()
    print(f"Loading params from ({path_to_checkpoint_file})...")
    params = paddle.load(path_to_checkpoint_file)
    model.set_dict(params)

    model.eval()

    input_spec = InputSpec(shape=[None, 3, 54, 54],
                           dtype='float32',
                           name='input'),
    model = to_static(model, input_spec=input_spec)
    paddle.jit.save(model, "inference")
    print(f"model (SVHN) has been already saved in (inference).")


def main(args):
    path_to_checkpoint_file = args.checkpoint

    _trans(path_to_checkpoint_file)


if __name__ == '__main__':
    main(parser.parse_args())
