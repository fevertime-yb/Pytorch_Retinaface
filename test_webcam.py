from __future__ import print_function

import argparse
import os

import cv2
import numpy as np
import torch
import yaml
import torch.backends.cudnn as cudnn

from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.timer import Timer
from utils.util import *
from pytorch_bn_fusion.bn_fusion import fuse_bn_recursively


parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument("--config", default="./configs/test_ResNet.yml",
                    help="Network config path")
parser.add_argument('--origin_size', action="store_true", default=False,
                    help='Whether use origin image size to evaluate')
parser.add_argument('--video-path', default=None, type=str,
                    help='input video path')
parser.add_argument("--bn_fusion", action="store_true",
                    help="Fusing model's convolution and batch norm layer")
args = parser.parse_args()


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # net and model
    net = RetinaFace(cfg=cfg, phase="test")
    net = load_model(net, cfg["model_path"], device)
    net.eval()
    cudnn.benchmark = True
    net = net.to(device)

    if args.bn_fusion:
        net = fuse_bn_recursively(net)

    im_height = cfg["im_height"] * 4 if args.origin_size else cfg["im_height"]
    im_width = cfg["im_width"] * 4 if args.origin_size else cfg["im_width"]

    print_FLOPS(net, input_size=(3, im_width, im_height))

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    scale = torch.tensor([im_width, im_height, im_width, im_height], dtype=torch.float, device=device)

    scale1 = torch.tensor([im_width, im_height, im_width, im_height,
                           im_width, im_height, im_width, im_height,
                           im_width, im_height], dtype=torch.float, device=device)

    _t = {"inference": Timer(), "post_process": Timer()}

    capture = cv2.VideoCapture(args.video_path if args.video_path else 0)
    cv2.namedWindow("VideoFrame", cv2.WINDOW_NORMAL)

    while capture.isOpened():
        ret, frame = capture.read()

        if not args.origin_size:
            frame = cv2.resize(frame, dsize=(im_width, im_height), interpolation=cv2.INTER_CUBIC)

        img = np.float32(frame)
        im_height, im_width, im_channel = img.shape

        img = toTensor(img, device=device, mean=(104, 117, 123))

        _t["inference"].tic()
        loc, conf, landmarks = net(img)
        _t["inference"].toc()

        _t["post_process"].tic()
        boxes = decode(loc.data.squeeze(0), prior_data, cfg["variance"])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landmarks = decode_landm(landmarks.data.squeeze(0), prior_data, cfg["variance"])
        landmarks = landmarks * scale1
        landmarks = landmarks.cpu().numpy()

        # ignore low scores
        idx = np.where(scores > cfg["confidence_threshold"])[0]
        boxes = boxes[idx]
        landmarks = landmarks[idx]
        scores = scores[idx]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:cfg["top_k"]]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        # do NMS
        detecting = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(detecting, cfg["nms_threshold"])
        detecting = detecting[keep, :]
        landmarks = landmarks[keep]

        # keep top-K faster NMS
        # detecting = detecting[:cfg["keep_top_k"], :]
        # landmarks = landmarks[:cfg["keep_top_k"], :]

        detecting = np.concatenate((detecting, landmarks), axis=1)
        _t["post_process"].toc()

        frame = draw_result(frame, detecting, cfg["vis_thresh"])

        cv2.imshow("VideoFrame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if _t["inference"].calls == 1000:
            break

    capture.release()
    cv2.destroyAllWindows()

    print("Model inference time: {:.4f} msec | NMS time: {:.4f} msec | images: {}".format(
        _t["inference"].average_time * 1000,
        _t["post_process"].average_time * 1000,
        _t["inference"].calls)
    )
