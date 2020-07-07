#!/usr/bin/env bash
docker run -it \
-v $PWD/data/:/test \
tfrecord \
--annotation_dir /test/annotations \
-l /test/label_map.pbtxt \
-o /test/train.record -s train \
--image_dir /test/images
