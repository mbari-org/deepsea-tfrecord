#!/usr/bin/env python

__author__ = "Danelle Cline"
__copyright__ = "Copyright 2020, MBARI"
__credits__ = ["MBARI"]
__license__ = "GPL"
__maintainer__ = "Danelle Cline"
__email__ = "dcline at mbari.org"
__doc__ = '''

Python test using the nose python framework

@author: __author__
@status: __status__
@license: __license__
'''
import sys
import os
import subprocess
from nose import with_setup
sys.path.append(os.path.join(os.path.dirname(__file__)))

print("")  # this is to get a newline after the dots
print("setup_module before anything in this file")
print('Setting up docker client')


def monitor(container):
    """
    Monitor running container and print output
    :param container:
    :return:
    """
    container.reload()
    l = ""
    while True:
        for line in container.logs(stream=True):
            l = line.strip().decode()
            print(l)
        else:
            break
    return l


def teardown_module(module):
    """
    Run after everything in this file completes
    :param module:
    :return:
    """
    print('tearing down')


def custom_setup_function():
    print("custom_setup_function")


def custom_teardown_function():
    print("custom_teardown_function")


@with_setup(custom_setup_function, custom_teardown_function)
def test_record():
    print('<============================ running test_record ============================ >')
    s = subprocess.check_call(
        ['python3', '/app/create_tfrecord.py', '--annotation_dir', '/test/annotations', '--image_dir',
         '/test/images', '--integer_id', 'True', '-l', '/test/label_map.pbtxt', '--set', 'test',
         '-o', '/test/test.record'], shell=False)
    print(s)
    # assert s == 'Image BGR mean [124.24453366 143.06497058  98.35049431] normalized [0.48723347 0.5610391  0.38568821]'
