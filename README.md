[MBARI](https://www.mbari.org/wp-content/uploads/2014/11/logo-mbari-3b.png)
<p align="right">
    <b> <img src="https://img.shields.io/badge/Supported%20Platforms-Windows%20%7C%20macOS%20%7C%20Linux-green" title="Supported Platforms"/> </b> <br>
    <b> <img src="https://img.shields.io/badge/license-GPL-blue" title="license-GPL"/> </b> <br>
</p>

# About

*deepsea-tfrecord* creates tensorflow records files from PNG images and PASCAL formatted annotations for use in the Tensorflow Object Detection API.
    
![ Image link ](/img/flow.jpg)

## *Arguments* 

  * -l name of TensorFlow label map file (extension .pbtxt)
  * --annotation_dir directory with annotations
  * -o path to output TFRecord (.record) file
  * --image_path directory with image data associated with annotations
  * --resize (optional) wxh resize images to wxh before storing in record file
  * --minsize (optional) wxh minimum size bounding box to include in record file
  * -s (optional) convert training or validation set (train/test)
  * --labels (optional) list of space separated labels to load. Defaults to everything listed in the label map proto. Names must exist in the label map proto file. 

## *Example*

Assuming data is stored in your current directory /data in the format
 
 * /data/imgs
 * /data/annotations
 * /data/label_map.pbtxt
 
 create tensorflow record using label map  /data/label_map.pbtxt on the data in /data and store in record file train.record.

```bash
docker run -it \
-v $PWD/data:/data \
-v  $PWD:/out \
mbari/deepsea-tfrecord \
-l /data/label_map.pbtxt \
--annotation_dir /data/annotations \
--image_dir /data/imgs \
-o /out/train.record \
-s train'
```

# Build for your own use
```bash
export DOCKER_GID=<your docker group id>
export DOCKER_UID=<your docker user id>
docker build --build-arg DOCKER_GID=$DOCKER_GID --build-arg DOCKER_UID=$DOCKER_UID -t tfrecord .
```

# References
https://www.tensorflow.org/tutorials/load_data/tfrecord
