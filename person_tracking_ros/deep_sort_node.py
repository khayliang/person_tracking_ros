import os
import cv2
import time
import torch
import warnings
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node

from geometry_msgs.msg import Point
from std_msgs.msg import Float32, Bool, Int32MultiArray, Int32
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from person_tracking_interfaces.srv import ChooseTarget, ClearTarget, UpdateTracker

from person_tracking_ros.deep_sort import build_tracker
from person_tracking_ros.utils.draw import draw_boxes

class VideoTracker(Node):
    def __init__(self, args):
        super().__init__('deep_sort')

        #declare params 
        #deepsort params
        self.declare_parameter('max_dist', 0.18)
        self.declare_parameter('min_confidence', 0.3)
        self.declare_parameter('nms_max_overlap', 0.5)
        self.declare_parameter('max_iou_distance', 0.2)
        self.declare_parameter('max_age', 1)
        self.declare_parameter('n_init', 3)
        self.declare_parameter('nn_budget', 100)
        self.declare_parameter('distance', 'cosine')
        self.declare_parameter('reid_ckpt', '')

        #detector params
        self.declare_parameter('classes', '')

        #tracker params
        self.declare_parameter('camera_fov', 60)
        self.declare_parameter('save_path', '')

        #misc params
        self.declare_parameter('frame_interval', 1)
        self.declare_parameter('save_results', False)
        self.declare_parameter('img_topic', '')
        self.declare_parameter('bbox_topic', '/detectnet/detections')
        
        #get params
        reid_ckpt = self.get_parameter('reid_ckpt').get_parameter_value().string_value
        classes_path = self.get_parameter('classes').get_parameter_value().string_value
        with open(classes_path, 'r', encoding='utf8') as fp:
            self.class_names = [line.strip() for line in fp.readlines()]
        
        save_path = self.get_parameter('save_path').get_parameter_value().string_value
        
        if not reid_ckpt or not save_path:
            raise Exception("Invalid or empty paths provided in YAML file.")
            
        deepsort_params = {
            "MAX_DIST": self.get_parameter('max_dist').get_parameter_value().double_value,
            "MIN_CONFIDENCE": self.get_parameter('min_confidence').get_parameter_value().double_value,
            "NMS_MAX_OVERLAP": self.get_parameter('nms_max_overlap').get_parameter_value().double_value,
            "MAX_IOU_DISTANCE": self.get_parameter('max_iou_distance').get_parameter_value().double_value,
            "MAX_AGE": self.get_parameter('max_age').get_parameter_value().integer_value,
            "N_INIT": self.get_parameter('n_init').get_parameter_value().integer_value,
            "NN_BUDGET": self.get_parameter('nn_budget').get_parameter_value().integer_value,
            "DISTANCE": self.get_parameter('distance').get_parameter_value().string_value,
            "REID_CKPT": reid_ckpt
        }

        args = {
            "frame_interval": self.get_parameter('frame_interval').get_parameter_value().integer_value,
            "save_results": self.get_parameter('save_results').get_parameter_value().bool_value,
            "img_topic": self.get_parameter('img_topic').get_parameter_value().string_value,
            "bbox_topic": self.get_parameter('bbox_topic').get_parameter_value().string_value
        }
        
        self.camera_fov = self.get_parameter("camera_fov").get_parameter_value().integer_value
        self.results_path = save_path

        #initialize publishers
        self.bbox_pub = self.create_publisher(Point, "/person_tracking/bbox_center", 1)
        self.angle_pub = self.create_publisher(Float32, "/person_tracking/target_angle", 1)

        self.target_present_pub = self.create_publisher(Bool, "/person_tracking/target_present", 1)
        self.target_indice_pub = self.create_publisher(Int32, "/person_tracking/target_indice", 1)

        self.detections_pub = self.create_publisher(Int32MultiArray, "/person_tracking/detection_indices", 1)

        self.image_pub = self.create_publisher(Image, "/person_tracking/deepsort_image", 1)

        #initialize services to interact with node
        self.target_clear_srv = self.create_service(ClearTarget, "/person_tracking/clear_target", self.clear_track)
        self.target_choose_srv = self.create_service(ChooseTarget, "/person_tracking/choose_target", self.select_target)

        self.cfg = {"DEEPSORT": deepsort_params}
        self.args = args

        self.logger = self.get_logger()
        self.bridge = CvBridge()
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        self.deepsort = build_tracker(self.cfg, use_cuda=use_cuda)
    
        #subscribes to image topic if topic name is given, else provide service if service argument is given
        if args["bbox_topic"] and args["img_topic"]:
            self.bbox_subscriber = self.create_subscription(Detection2DArray, args["bbox_topic"], self.ros_deepsort_callback, 1)
            self.img_subscriber = self.create_subscription(Image, args["img_topic"], self.img_callback, 1)
        else:
            raise Exception("no topic given. Ending node.")

        self.idx_frame = 0
        self.idx_tracked = None
        self.bbox_xyxy = []
        self.identities = []
        self.img_msg = None

        self.writer = None

        self.logger.info('Video stream has started')

    def __del__(self):
        if self.writer:
            self.writer.release()

    #callback function to clear track
    def clear_track(self, request, response):
        if self.idx_tracked is not None and request.clear:
            self.idx_tracked = None
            response.success = True
        else:
            response.success = False
        return response

    #callback function to select target
    def select_target(self, ros_data, response):
        if self.idx_tracked is None:
            for identity in self.identities:
                if identity == ros_data.target:
                    self.idx_tracked = ros_data.target
                    response.success = True
                    return response
            
            response.success = False
            return response
        else:
            response.success = False
            return response
        return response

    def img_callback(self, msg):
        self.img_msg = msg

    #main deepsort callback function
    def ros_deepsort_callback(self, msg):
        
        start = time.time()

        if self.img_msg == None:
            return

        #convert ros Image message to opencv
        ori_im = self.bridge.imgmsg_to_cv2(self.img_msg, "bgr8")  
        im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

        #skip frame per frame interval
        self.idx_frame += 1
        if self.idx_frame % self.args["frame_interval"]:
            return
            
        # parse ros message
        bbox_xywh = []
        cls_conf = []
        cls_ids = []
        for detection in msg.detections:
            bbox = detection.bbox
            bbox_xywh.append([bbox.center.x, bbox.center.y, bbox.size_x, bbox.size_y])

            class_info = detection.results[0]
            cls_conf.append(class_info.score)
            # convert ascii to int because ros_deep_learning gives id in char
            cls_ids.append(ord(class_info.id))

        bbox_xywh = np.array(bbox_xywh)
        cls_conf = np.array(cls_conf)
        cls_ids = np.array(cls_ids)
        
        # select person class
        mask = cls_ids==1
        bbox_xywh = bbox_xywh[mask]
        # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
        bbox_xywh[:,3:] *= 1.2 
        cls_conf = cls_conf[mask]

        # do tracking
        outputs = self.deepsort.update(bbox_xywh, cls_conf, im, tracking_target=self.idx_tracked)

        # if detection present draw bounding boxes
        if len(outputs) > 0:
            bbox_tlwh = []
            self.bbox_xyxy = outputs[:,:4]
            # detection indices
            self.identities = outputs[:,-1]
            ori_im = draw_boxes(ori_im, self.bbox_xyxy, self.identities)

            for bb_xyxy in self.bbox_xyxy:
                bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

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
        
        # Publish new image
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(ori_im, "bgr8"))

        if self.args["save_results"]:
            if not self.writer:
                os.makedirs(self.results_path, exist_ok=True)

                # path of saved video and results
                save_video_path = os.path.join(self.results_path, "results.avi")

                # create video writer
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.writer = cv2.VideoWriter(save_video_path, fourcc, 20, tuple(ori_im.shape[1::-1]))
            
            self.writer.write(ori_im)

        # logging
        self.logger.info("frame: {}, time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                        .format(self.idx_frame, end-start, 1/(end-start), bbox_xywh.shape[0], len(outputs)))
    
        """publishing to topics"""
        #publish detection identities
        self.detections_pub.publish(Int32MultiArray(data=self.identities))
        
        #publish if target present
        if len(outputs) == 0 or self.idx_tracked is None:
            self.target_present_pub.publish(Bool(data=False))
        elif len(outputs) > 0 and self.idx_tracked:
            self.target_present_pub.publish(Bool(data=True))

        #publish angle and xy data
        if self.idx_tracked is not None:

            x_center = (self.bbox_xyxy[0][0] + self.bbox_xyxy[0][2])/2
            y_center = (self.bbox_xyxy[0][1] + self.bbox_xyxy[0][3])/2

            pixel_per_angle = im.shape[1]/self.camera_fov

            x_center_adjusted = x_center - (im.shape[1]/2)

            angle = x_center_adjusted/pixel_per_angle

            self.bbox_pub.publish(Point(x=float(x_center), y=float(y_center), z=0.0))
            self.angle_pub.publish(Float32(data=angle))
            self.target_indice_pub.publish(Int32(data=self.idx_tracked))
        else:
            self.target_indice_pub.publish(Int32(data=-1))
            self.bbox_pub.publish(Point(x=0.0, y=0.0, z=0.0))

def main(args=None):
    '''Initializes and cleanup ros node'''
    rclpy.init()
    person_tracker = VideoTracker(args)
    
    rclpy.spin(person_tracker)

    person_tracker.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
