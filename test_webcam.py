from __future__ import print_function

import argparse
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.timer import Timer
from utils.util import *


parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', type=str,
                    choices=["mobile0.25", "resnet50"],
                    help="Backbone network mobile0.25 or resnet50")
parser.add_argument('--origin_size', action="store_true", default=False,
                    help='Whether use origin image size to evaluate')
parser.add_argument('--confidence_threshold', default=0.02, type=float,
                    help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int,
                    help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float,
                    help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int,
                    help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False,
                    help='show detection results')
parser.add_argument('--vis_thresh', default=0.5, type=float,
                    help='visualization_threshold')
parser.add_argument('--video-path', default=None, type=str,
                    help='input video path')
args = parser.parse_args()


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # net and model
    net = RetinaFace(cfg=cfg, phase="test")
    net = load_model(net, args.trained_model, device)
    net.eval()
    cudnn.benchmark = True
    net = net.to(device)

    _t = {"inference": Timer(), "post_process": Timer()}

    print_params_FLOPS(net)

    im_height = 270
    im_width = 480

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    scale = torch.tensor([im_width, im_height, im_width, im_height], dtype=torch.float, device=device)

    scale1 = torch.tensor([im_width, im_height, im_width, im_height,
                           im_width, im_height, im_width, im_height,
                           im_width, im_height], dtype=torch.float, device=device)

    if not args.video_path:
        raise RuntimeError("check video path")

    capture = cv2.VideoCapture(args.video_path)
    cv2.namedWindow("VideoFrame", cv2.WINDOW_NORMAL)

    while capture.isOpened():
        ret, frame = capture.read()
        if args.origin_size:
            frame = cv2.resize(frame, dsize=(im_width, im_height), interpolation=cv2.INTER_CUBIC)
        img = np.float32(frame)
        im_height, im_width, im_channel = img.shape

        # normalize input tensor
        img = toTensor(img, device=device, mean=(104, 117, 123))

        _t["inference"].tic()
        loc, conf, landmarks = net(img)
        _t["inference"].toc()

        _t["post_process"].tic()
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landmarks = decode_landm(landmarks.data.squeeze(0), prior_data, cfg['variance'])
        landmarks = landmarks * scale1
        landmarks = landmarks.cpu().numpy()

        # ignore low scores
        idx = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[idx]
        landmarks = landmarks[idx]
        scores = scores[idx]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        # do NMS
        detecting = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(detecting, args.nms_threshold)
        detecting = detecting[keep, :]
        landmarks = landmarks[keep]

        # keep top-K faster NMS
        # detecting = detecting[:args.keep_top_k, :]
        # landmarks = landmarks[:args.keep_top_k, :]

        detecting = np.concatenate((detecting, landmarks), axis=1)
        _t["post_process"].toc()

        frame = draw_result(frame, detecting, args.vis_thresh)

        cv2.imshow("VideoFrame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

    print("inference time: {:.4f}s | nms time: {:.4f}s".format(
        _t["inference"].average_time,
        _t["post_process"].average_time)
    )
