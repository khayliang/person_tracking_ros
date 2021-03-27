#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$parent_path"

ros2 run person_tracking_ros deep_sort --ros-args --params-file ../configs/deep_sort.yaml
