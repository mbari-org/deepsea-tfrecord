#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Creates tensorflow records for use in box proposal detection networks

@author: __author__
@status: __status__
@license: __license__
'''

import io
import glob
import os
import numpy as np
import logging
import sys
import cv2
import random

sys.path.append(os.path.join(os.path.dirname(__file__), 'tensorflow_models/research'))
sys.path.append(os.path.join(os.path.dirname(__file__)))
if __package__ == None:
    __version__ = "1.0.0"
else:
    from . import __version__


from lxml import etree
import tensorflow as tf
import hashlib
import tempfile
from PIL import Image

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_detection.core import standard_fields

SETS = ['train', 'val', 'trainval', 'test']

TARGET_WIDTH = 960
TARGET_HEIGHT = 540


def process_command_line():
    """
    Process command line
    :return: args object
    """
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    import argparse
    from argparse import RawTextHelpFormatter

    examples = 'Examples:' + '\n\n'
    examples += 'Create record for xml files in /data:\n'
    examples += '{0} --annotation_dir /data --image_dir /data --output_record MBARI_BENTHIC_2017_test.record ' \
                '   --label_map_path /data/mbari_benthic_label_map.pbtxt' \
                '--set test '.format(sys.argv[0])
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                     description='Creates Tensorflow Record object for MBARI annotated data',
                                     epilog=examples)
    parser.add_argument('--annotation_dir', action='store', help='Directory with annotations', required=True)
    parser.add_argument('--image_dir', action='store', help='Directory with images associated with annotations',
                        default='', required=True)
    parser.add_argument('-o', '--output_record', action='store', help='Path to output TFRecord', required=True)
    parser.add_argument('-l', '--label_map_path', action='store', help='Path to label map proto', required=False)
    parser.add_argument('-s', '--set', action='store', help='Convert training set, validation set or merged set.',
                        required=False)
    parser.add_argument('--version', help='Print version', default=__version__)
    parser.add_argument('--split', help='Train/test split.', required=False, type=str, default="0.8, 0.2")
    parser.add_argument('--resize', help='Resize images to wxh', required=False, type=str, default="960x540")
    parser.add_argument('--minsize', help='Minimum size bounding box to include in record wxh', required=False,
                        type=str, default="75x75")
    parser.add_argument('--grayscale', help='Convert images to grayscale', type=str2bool, default=False)
    parser.add_argument('--integer_id', help='Convert images and annotations to integer ids as required for COCO ',
                        type=str2bool, default=False)
    parser.add_argument('--deinterlace', help='Deinterlace', type=str2bool, default=False)
    parser.add_argument('--labels', action='store',
                        help='List of space separated labels to load. Must be in the label map proto', nargs='*',
                        required=False)

    args = parser.parse_args()
    return args


def split(labels, output_dir, train_per, test_per):
    '''
    Split annotations into train/test in a particular output_dir
    Outputs the split in two files named train.txt and test.txt
    :param output_dir:
    :param train_per:
    :param test_per:
    :return:
    '''
    annotations = []
    print('Searching in {}'.format(output_dir))
    for xml_in in sorted(glob.glob(output_dir + '/*.xml')):
        if is_valid_xml(xml_in, labels) == True:
            annotations.append(xml_in)

    print('Found {} xml annotations in {}'.format(len(annotations), output_dir))

    # split randomly
    random.shuffle(annotations)
    total = len(annotations)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(annotations) > 0:
        with open(os.path.join(output_dir, 'train.txt'), 'wt') as f:
            for i in range(int(round(train_per * total))):
                f.write("{}\n".format(annotations[i]))

        with open(os.path.join(output_dir, 'test.txt'), 'wt') as f:
            for i in range(int(round(test_per * total))):
                f.write("{}\n".format(annotations[-1 * i]))


def is_valid_xml(xml_filename, labels):
    try:
        with open(xml_filename, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        for obj in data['object']:
            name = obj['name']
            if labels and name in labels:
                return True
            else:
                continue
    except Exception as ex:
        return False
    return False


def resize(image_dir, target_width, target_height, deinterlace=False, grayscale=False):
    img = cv2.imread(image_dir)
    h, w, channels = img.shape
    print('Reading {} size {}x{} rescaling to {}x{}'.format(image_dir, w, h, target_width, target_height))

    fd, path = tempfile.mkstemp(suffix='.png')
    try:
        if deinterlace and h == 2 * target_height and w == 2 * target_width:
            print('Deinterlacing {} {}x{} to {}x{}'.format(image_dir, h, w, target_height, target_width))
            final_image = img[::2, 1::2]
        elif h != target_height or w != target_width:
            print('Rescaling {} {}x{} to {}x{}'.format(image_dir, w, h, target_width, target_height))
            final_image = cv2.resize(img, (target_width, target_height))
        else:
            final_image = img

        if grayscale:
            cv2.imwrite(path, cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY))
        else:
            cv2.imwrite(path, final_image)

        mean = np.mean(final_image, axis=(0, 1))
        std = np.std(final_image, axis=(0, 1))
        with tf.gfile.GFile(path, 'rb') as fid:
            encoded_png = fid.read()
    finally:
        os.remove(path)

    return encoded_png, mean, std


def img_to_tf(img_path, width, height, deinterlace, grayscale):
    """
    Convert image to tensorflow record
    :param img_path: full path to image
    :param width: width to resize image to
    :param height: height to resize image to
    :param deinterlace:  true if deinterlacing needed
    :param grayscale:  to true if convert to grayscale
    :return: record, mean of the image
    """
    _, filename = os.path.split(img_path)
    encoded_png, mean, std = resize(img_path, width, height, deinterlace, grayscale)
    encoded_png_io = io.BytesIO(encoded_png)
    image = Image.open(encoded_png_io)
    if image.format != 'PNG':
        raise ValueError('Image format not PNG')
    png_key = hashlib.sha256(encoded_png).hexdigest()
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(png_key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
        'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return example, mean, std


def dict_to_tf_example(id,
                       data,
                       image_dir,
                       label_map_dict,
                       labels,
                       width,
                       height,
                       minsize,
                       deinterlace,
                       grayscale):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    :param id: image identifier to use in record
    :param data: dict holding XML fields for a single image (obtained by
    :param running dataset_util.recursive_parse_xml_to_dict)
    :param image_dir: image directory
    :param running dataset_util.recursive_parse_xml_to_dict)
    :param label_map_dict: A map from string label names to integers ids.
    :param labels: list of labels to include in the record
    :param width: width to resize images to
    :param height: height to resize images to
    :param minsize: minimum size in pixels to include in record
    :param deinterlace:  true if deinterlacing needed
    :param grayscale:  to true if convert to grayscale

    Returns:
      example: The converted tf.Example, label dictionary and the mean RGB value for this example

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid PNG
    """

    img_path = os.path.join(image_dir, data['filename'])
    encoded_png, mean, std = resize(img_path, width, height, deinterlace, grayscale)
    encoded_png_io = io.BytesIO(encoded_png)
    image = Image.open(encoded_png_io)
    if image.format != 'PNG':
        raise ValueError('Image format not PNG')
    png_key = hashlib.sha256(encoded_png).hexdigest()

    scale_width = float(width) / float(data['size']['width'])
    scale_height = float(height) / float(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    my_labels = dict.fromkeys(label_map_dict.keys(), 0)
    width_max = int(minsize.split('x')[0])
    height_max = int(minsize.split('x')[1])

    for obj in data['object']:
        name = obj['name']
        print('Converting {} {}'.format(name, data))
        if labels and name not in labels:
            print('{} not in {} so excluding from record'.format(name, labels))
            continue

        xmn = int(obj['bndbox']['xmin'])
        ymn = int(obj['bndbox']['ymin'])
        xmx = int(obj['bndbox']['xmax'])
        ymx = int(obj['bndbox']['ymax'])

        # skip over small objects
        if abs(xmx - xmn) < width_max and abs(ymx - ymn) < height_max:
            print('{} smaller than {} pixels so excluding from record'.format(name, minsize))
            continue

        xmin.append(float(xmn) * scale_width / width)
        ymin.append(float(ymn) * scale_height / height)
        xmax.append(float(xmx) * scale_width / width)
        ymax.append(float(ymx) * scale_height / height)
        classes_text.append(name.encode('utf8'))
        if name not in label_map_dict.keys():
            print('{} not in label map so skipping'.format(name))
            continue
        key = label_map_dict[name]
        classes.append(key)
        my_labels[name] += 1

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            id.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            id.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(png_key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
        'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example, my_labels, mean, std


def main(_):
    args = process_command_line()

    if args.set and args.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    if args.resize:
        width = int(args.resize.split('x')[0])
        height = int(args.resize.split('x')[1])
    else:
        width = TARGET_WIDTH
        height = TARGET_HEIGHT
    print('Rescaling images to {}x{}'.format(width, height))
    output = os.path.join(args.annotation_dir, args.output_record)

    # touch the file if it doesn't already exist
    if not os.path.exists(output):
        with open(output, 'a'):
            os.utime(output, None)

    writer = tf.python_io.TFRecordWriter(output)

    label_map_dict = {}
    label_example = {}
    has_labels = False
    means = []
    stds = []

    if args.label_map_path:
        d = label_map_util.get_label_map_dict(args.label_map_path)
        for key, value in d.items():
            label_map_dict[key] = value
        labels = dict.fromkeys(label_map_dict.keys(), 0)
        has_labels = True

    if has_labels and args.set:
        examples_path = os.path.join(args.annotation_dir, args.set + '.txt')
        # do the split if the file doesn't exist
        if not os.path.exists(examples_path):
            my_split = [float(item) for item in args.split.split(',')]
            train_per = my_split[0]
            tests_per = my_split[1]
            print('Splitting train and testing data {} {}'.format(train_per, tests_per))
            split(labels, args.annotation_dir, train_per, tests_per)

        print('Reading examples from {}'.format(examples_path))
        if os.path.exists(examples_path):
            with open(examples_path) as fid:
                lines = fid.readlines()
                examples_list = [line.strip() for line in lines]
        else: 
            examples_list = []

        for idx, example in enumerate(examples_list):
            if idx % 10 == 0:
                print('Processing image {} of {}'.format(idx, len(examples_list)))
            try:
                if not example.endswith('.xml'):
                    example += '.xml'
                example = os.path.join(args.annotation_dir, example)
                with open(example, 'r') as fid:
                    xml_str = fid.read()
                    xml = etree.fromstring(xml_str)
                    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
                    if args.integer_id:
                        id = str(idx)
                        print('<============================================================>')
                    else:
                        id = data['filename']
                    tf_example, label_example, mean, std = dict_to_tf_example(id, data, args.image_dir, label_map_dict,
                                                                         args.labels,
                                                                         width, height, args.minsize, args.deinterlace,
                                                                         args.grayscale)
                    means.append(mean)
                    stds.append(std)
                    if tf_example:
                        for key, value in label_example.items():
                            labels[key] += value
                        writer.write(tf_example.SerializeToString())
            except Exception as ex:
                print(ex)
                continue
            else:
                logging.warning('No objects found in {}'.format(example))

        writer.close()
        mean = 0
        std = 0
        if len(means) > 0 and len(stds) > 0:
            mean = np.mean(means, axis=(0))
            std = np.mean(stds, axis=(0))

        for key, value in labels.items():
            print('Total {} = {}'.format(key, value))

        print('Done. Found {} examples in {} set.\nImage BGR mean {} normalized {} std {} normalized {}'.
              format(sum(labels.values()), args.set, mean, mean / 255, std, std / 255 ))
    else:
        in_path = os.path.join(args.image_dir, '*.png')
        print('Searching for examples in {}'.format(in_path))
        examples_list = sorted(glob.glob(in_path))
        for example in examples_list:
            try:
                tf_example, mean = img_to_tf(example, width, height, args.deinterlace, args.grayscale)
                means.append(mean)
                if tf_example and has_labels:
                    for key, value in label_example.items():
                        labels[key] += value
                    writer.write(tf_example.SerializeToString())
            except Exception as ex:
                print(ex)
                continue

        writer.close()
        mean = 0
        std = 0
        if len(means) > 0 and len(stds) > 0:
            mean = np.mean(means, axis=(0))
            std = np.mean(stds, axis=(0))

        print('Done. Found {} examples. \nImage BGR mean {} normalized {} std {} normalized {}'.
              format(len(examples_list), mean, mean / 255, std, std / 255))

if __name__ == '__main__':
    tf.app.run()
