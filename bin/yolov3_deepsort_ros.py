#!/home/kl/torch_gpu_ros/bin/python

import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
import rospy
import sys

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results

from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image

class VideoTracker(object):
    def __init__(self, cfg, args, video_path):

        rospy.init_node('coordinate_publisher')
        rospy.loginfo('Video stream has started')

        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names


    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width,self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        idx_tracked = None
        bbox_xyxy = []

        bbox_pub = rospy.Publisher("/bbox_center", Point, queue_size=10)
        angle_pub = rospy.Publisher("/target_angle", Float32, queue_size=10)

        while self.vdo.grab() and not rospy.is_shutdown():

            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            if idx_frame < args.load_from:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)

            # select person class
            mask = cls_ids==0

            bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            bbox_xywh[:,3:] *= 1.2 
            cls_conf = cls_conf[mask]

            # do tracking
            if idx_tracked:
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im, tracking_target=idx_tracked)
            else:
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

            #idx_tracked = 0

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:,:4]
                identities = outputs[:,-1]
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                #print(bbox_xyxy)
                #print(idx_tracked)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame-1, bbox_tlwh, identities))

            end = time.time()

            #draw frame count
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,500)
            fontScale              = 1
            fontColor              = (255,255,255)
            lineType               = 2
            frame_count = ("Frame no: %d" % idx_frame)
            cv2.putText(ori_im,frame_count, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            #draw tracking number
            if idx_tracked:
                tracking_str = ("Tracking: %d" % idx_tracked)
            else:
                tracking_str = ("Tracking: None")

            bottomLeftCornerOfText = (10,550)
            cv2.putText(ori_im,tracking_str, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)

            #get user input on target to track
            if self.args.display:
                cv2.imshow("test", ori_im)
                if cv2.waitKey(1) == ord('i'):
                    print("\nEnter target number for constant tracking")
                    user_input = input()

                    idx_tracked = int(user_input)      
                    
            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("frame: {}, time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                            .format(idx_frame, end-start, 1/(end-start), bbox_xywh.shape[0], len(outputs)))
            
            #publishing 
            if idx_tracked is not None:

                x_center = (bbox_xyxy[0][0] + bbox_xyxy[0][2])/2
                y_center = (bbox_xyxy[0][1] + bbox_xyxy[0][3])/2

                fov = 60

                pixel_per_angle = im.shape[1]/fov

                x_center_adjusted = x_center - (im.shape[1]/2)
                print(x_center_adjusted)
                angle = x_center_adjusted/pixel_per_angle
                print(angle)

                bbox_pub.publish(x_center,y_center,0)
                angle_pub.publish(angle)

                rate = rospy.Rate(20) #10Hz

                rate.sleep()

        rospy.loginfo("Stopped sending bounding boxes")


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="./Jiang_family.mp4")
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument("--load_from", type=int, default=1)

    return parser.parse_known_args()

def main(args):
    '''Initializes and cleanup ros node'''
    person_tracker = VideoTracker()
    rospy.init_node('person_tracker', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS tracker")


if __name__=="__main__":

    print("\n\nyeett\n\n")
    
    args, unknown = parse_args()
    cfg = get_config()
    print(args.config_detection)
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=args.video_path) as vdo_trk:
        vdo_trk.run()
