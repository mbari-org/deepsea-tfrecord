#!/usr/bin/env bash
docker build --build-arg DOCKER_GID=`id -g` --build-arg DOCKER_UID=`id -u` -t mbari/deepsea-tfrecord .
