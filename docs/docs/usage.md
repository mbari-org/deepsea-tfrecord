## *Arguments* 

| Argument | Description |
|---|---|
|-l |name of TensorFlow label map file (extension .pbtxt)|
|--annotation_dir| directory with annotations|
|-o |path to output TFRecord (.record) file|
|--image_path [directory] |directory with image data associated with annotations|
|--resize [wxh] |(optional) resize images to wxh before storing in record file, e.g. --resize 512x512 resizes to 512x512 pixels|
|--minsize [wxh] | (optional) wxh minimum size bounding box to include in record file|
|-s [train or test] | (optional) convert training or validation set (train/test)|
|--labels [list of labels] | (optional) list of space separated labels to load. Defaults to everything listed in the label map proto. Names must exist in the label map proto file. |

## *Example*

Assuming data is stored in your current directory in the layout
 
~~~
│   └── imgs
│   └── annotations
│   └── label_map.pbtxt
~~~

Create tensorflow record using label map  /data/label_map.pbtxt on the data in /data and store in record file train.record.
 
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

 * run as you **-u $(id -u):$(id -g)**
 * remove after running **--rm**
 * run interactively **-it**
 * mount your current directory to /data **-v $PWD:/data**
