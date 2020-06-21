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
from std_msgs.msg import Float32, Bool, Int32MultiArray, Int32
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import CompressedImage

"""
args: display, img_topic, camera, save_path, video_path frame_interval

"""

class VideoTracker(object):
    def __init__(self, args):

        #get rosparams 
        yolov3_params = rospy.get_param("/YOLOV3")
        deepsort_params = rospy.get_param("/DEEPSORT")
        
        self.camera_fov = rospy.get_param("person_tracker/camera_fov")

        #initialize ros node to person_tracker name
        rospy.init_node('person_tracker', anonymous=True)
        rospy.loginfo('Video stream has started')

        #initialize publishers
        self.bbox_pub = rospy.Publisher("/person_tracking/bbox_center", Point, queue_size=1)
        self.angle_pub = rospy.Publisher("/person_tracking/target_angle", Float32, queue_size=1)

        self.target_present_pub = rospy.Publisher("/person_tracking/target_present", Bool, queue_size=1)
        self.target_started_pub = rospy.Publisher("/person_tracking/track_started", Bool, queue_size=1)

        self.detections_pub = rospy.Publisher("person_tracking/detection_indices", Int32MultiArray, queue_size=1)

        self.image_pub = rospy.Publisher("/person_tracking/deepsort_image/compressed", CompressedImage)

        #initialize subscribers to interact with node
        self.target_clear_sub = rospy.Subscriber("person_tracking/clear_target", Bool, self.clear_track, queue_size=1)
        self.choose_target_sub = rospy.Subscriber("person_tracking/choose_target", Int32, self.select_target, queue_size=1)

        #concatenate rosparams to yolov3_deepsort style
        self.cfg = {"YOLOV3": yolov3_params, "DEEPSORT": deepsort_params}
        self.args = args

        #TODO:default processing if no topic or webcam given: source is jiang_fam
        if args.video_path:
            self.video_path = args.video_path

        self.logger = get_logger("root")

        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        self.detector = build_detector(self.cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(self.cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names
        
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        #subscribes to comprossed image topic if topic name is given
        if args.img_topic:
            self.img_subscriber = rospy.Subscriber(args.img_topic, CompressedImage, self.ros_deepsort_callback,  queue_size = 1, buff_size=2**24)
        #check if webcam is given if comprssed image topic not given        
        elif args.camera != -1:
            print("Using webcam " + str(args.camera))
            self.vdo = cv2.VideoCapture(args.camera)
        #use video path
        else:
            self.vdo = cv2.VideoCapture()

        self.results = []
        self.idx_frame = 0
        self.idx_tracked = None
        self.bbox_xyxy = []

    def __enter__(self):
        if self.args.camera != -1:
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

    #callback function to clear track
    def clear_track(self, ros_data):
        try:
            if self.idx_tracked is not None and ros_data.data:
                if self.idx_tracked == -1:
                    self.idx_tracked = None
                elif self.idx_tracked is not None:
                    self.idx_tracked = -1
        except:
            return

    #callback function to select target
    def select_target(self, ros_data):
        try:
            if self.idx_tracked is None:
                self.idx_tracked = ros_data.data
        except:
            return
        

    def ros_deepsort_callback(self, ros_data):

        start = time.time()

        #convert ros compressed image message to opencv 
        np_arr = np.fromstring(ros_data.data, np.uint8)

        ori_im = cv2.imdecode(np_arr, flags=cv2.IMREAD_COLOR)
        im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

        #skip frame per frame interval
        self.idx_frame += 1
        if self.idx_frame % self.args.frame_interval:
            return

        # do detection
        bbox_xywh, cls_conf, cls_ids = self.detector(im)

        # select person class
        mask = cls_ids==0

        bbox_xywh = bbox_xywh[mask]
        # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
        bbox_xywh[:,3:] *= 1.2 
        cls_conf = cls_conf[mask]

        # do tracking
        if self.idx_tracked:
            #tracking if index for target is selected
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im, tracking_target=self.idx_tracked)
        else:
            #tracking if not selected
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

        # if detection present draw bounding boxes
        identities = []
        if len(outputs) > 0:
            bbox_tlwh = []
            self.bbox_xyxy = outputs[:,:4]
            # detection indices
            identities = outputs[:,-1]
            ori_im = draw_boxes(ori_im, self.bbox_xyxy, identities)

            for bb_xyxy in self.bbox_xyxy:
                bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

            self.results.append((self.idx_frame-1, bbox_tlwh, identities))

        end = time.time()

        #draw frame count
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
        frame_count = ("Frame no: %d" % self.idx_frame)
        cv2.putText(ori_im,frame_count, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        #draw tracking number
        if self.idx_tracked:
            tracking_str = ("Tracking: %d" % self.idx_tracked)
        else:
            tracking_str = ("Tracking: None")

        bottomLeftCornerOfText = (10,550)
        cv2.putText(ori_im,tracking_str, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        
        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', ori_im)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)

        if self.args.save_path:
            self.writer.write(ori_im)
            # save results
            write_results(self.save_results_path, results, 'mot')

        # logging
        self.logger.info("frame: {}, time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                        .format(self.idx_frame, end-start, 1/(end-start), bbox_xywh.shape[0], len(outputs)))
    



        #publishing to topics

        #publish detection identities
        identity_msg = Int32MultiArray(data=identities)
        self.detections_pub.publish(identity_msg)

        #publish if target present
        if len(outputs) == 0 or self.idx_tracked is None:
            self.target_present_pub.publish(Bool(False))
        elif len(outputs) > 0 and self.idx_tracked:
            self.target_present_pub.publish(Bool(True))


        #publish angle and xy data
        if self.idx_tracked is not None:

            x_center = (self.bbox_xyxy[0][0] + self.bbox_xyxy[0][2])/2
            y_center = (self.bbox_xyxy[0][1] + self.bbox_xyxy[0][3])/2

            pixel_per_angle = im.shape[1]/self.camera_fov

            x_center_adjusted = x_center - (im.shape[1]/2)
            #print(x_center_adjusted)
            angle = x_center_adjusted/pixel_per_angle
            #print(angle)

            self.bbox_pub.publish(x_center,y_center,0)
            self.angle_pub.publish(angle)
            self.target_started_pub.publish(Bool(True))
        #publish if target initialized
        else:
            self.target_started_pub.publish(Bool(False))
    #TODO: function for if no ros topic
   # def run(self):

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--camera", type=int, default="-1")
    parser.add_argument("--img_topic", type=str)

    return parser.parse_known_args()
            
def main(args):
    '''Initializes and cleanup ros node'''
    person_tracker = VideoTracker(args)
    rospy.init_node('person_tracker', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS tracker")


if __name__=="__main__":
    args, unknown = parse_args()
    main(args)