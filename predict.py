import argparse
import paddle

from PIL import Image
from paddle.vision import transforms
from paddle.inference import Config, create_predictor
from model import Model
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, required=True, help='path to checkpoint, e.g. ./logs/model-100.pdparams')
parser.add_argument('--input', type=str, help='path to input image')


def _infer_static(model_file, params_file, input_path):
    config = Config(model_file, params_file)
    config.enable_use_gpu(8000, 0)
    config.switch_ir_optim(True)
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_tensor_list = []
    output_tensor_list = []
    for item in input_names:
        input_tensor_list.append(predictor.get_input_handle(item))
    for item in output_names:
        output_tensor_list.append(predictor.get_output_handle(item))
    outputs = []

    transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.CenterCrop([54, 54]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image = Image.open(input_path)
    image = image.convert('RGB')
    image = transform(image)
    images = image.unsqueeze(axis=0)
    images = images.numpy()
    for i in range(len(input_tensor_list)):
        input_tensor_list[i].copy_from_cpu(images)
    predictor.run()
    for j in range(len(output_tensor_list)):
        outputs.append(output_tensor_list[j].copy_to_cpu())
    for j in range(len(outputs)):
        outputs[j] = np.argmax(outputs[j], 1)
    print(outputs)

# length: 2
# digits: 7 5 10 10 10
def main(args):
    pdmodel = args.p
    pdiparams = args.p.replace('pdmodel', 'pdiparams')
    input_path = args.input
    _infer_static(pdmodel, pdiparams, input_path)


if __name__ == '__main__':
    main(parser.parse_args())
