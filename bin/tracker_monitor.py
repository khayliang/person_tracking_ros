class TrackerMonitor(object):
    def __init__:
        self.new_frame 

def main(args):
    rospy.init_node('tracker_monitor', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS tracker")

if __name__=="__main__":
    args, unknown = parse_args()
    main(args)