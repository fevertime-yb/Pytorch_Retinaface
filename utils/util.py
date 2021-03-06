import cv2
import numpy as np
import torch
from ptflops import get_model_complexity_info


def toTensor(img, device, mean=(128, 128, 128)):

    img -= mean
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    return img


def print_params_FLOPS(model, input_size=(3, 640, 480)):

    macs, params = get_model_complexity_info(model=model,
                                             input_res=input_size,
                                             as_strings=True,
                                             print_per_layer_stat=False,
                                             verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def load_model(model, model_path, device):
    print("Loading pre-trained model from {}".format(model_path))

    ckpt = torch.load(model_path, map_location=device)
    trained_dict = ckpt["state_dict"] if "state_dict" in ckpt.keys() else ckpt
    trained_dict = remove_prefix(trained_dict, "module.")
    check_keys(model, trained_dict)
    model.load_state_dict(trained_dict, strict=False)

    return model


def check_keys(model, ckpt):
    ckpt_keys = set(ckpt.keys())
    model_keys = set(model.state_dict().keys())

    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('Missing keys: {}'.format(len(missing_keys)))
    print('Unused checkpoint keys: {}'.format(len(unused_pretrained_keys)))
    print('Used keys: {}'.format(len(used_pretrained_keys)))

    assert len(used_pretrained_keys) > 0, 'load NONE from pre-trained checkpoint'

    return True


def remove_prefix(state_dict, prefix):
    """
    Old style model is stored with all names of parameters sharing common prefix 'module.'
    """
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def draw_result(image, detecting, thresh):

    for b in detecting:
        if b[4] < thresh:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))

        # Draw face rectangle
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

        # Print accuracy
        cv2.putText(image, text, (b[0], b[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # Draw face landmark
        cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(image, (b[13], b[14]), 1, (255, 0, 0), 4)

    return image
