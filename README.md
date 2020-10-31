[![MBARI](https://www.mbari.org/wp-content/uploads/2014/11/logo-mbari-3b.png)](http://www.mbari.org)

[![semantic-release](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--release-e10079.svg)](https://github.com/semantic-release/semantic-release)
![Supported Platforms](https://img.shields.io/badge/Supported%20Platforms-Windows%20%7C%20macOS%20%7C%20Linux-green)
![license-GPL](https://img.shields.io/badge/license-GPL-blue)

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

Assuming data is stored in your current directory in the layout
 
 * imgs/
 * annotations/
 * label_map.pbtxt
 
 create tensorflow record using label map  /data/label_map.pbtxt on the data in /data and store in record file train.record.

 * run as you -u $(id -u):$(id -g)
 * remove after running --rm
 * run interactively -it
 * mount your current directory to /data -v $PWD:/data

```bash
docker run -it --rm \
-v $PWD/data:/data \
-v  $PWD:/out \
-u $(id -u):$(id -g) \
mbari/deepsea-tfrecord \
-l /data/label_map.pbtxt \
--annotation_dir /data/annotations \
--image_dir /data/imgs \
-o /out/train.record \
-s train
```

# Build 
```bash
./build.sh
```

# References
https://www.tensorflow.org/tutorials/load_data/tfrecord
