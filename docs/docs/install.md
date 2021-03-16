# Installation

## Requirement
[Docker](http://docker.com/)

A docker images of this is available in hub.docker.com, or if you prefer to build yourself a script is provided 
for that:
 
```bash
./build.sh
```

This simply runs the docker build command for you:

```
docker build -t mbari/deepsea-tfrecord .
```

Once this is complete you should see the docker image mbari/deepsea-tfrecord

```
docker images | grep mbari/deepsea-tfrecord
```
```
REPOSITORY               TAG                 IMAGE ID            CREATED             SIZE
mbari/deepsea-tfrecord   latest              2e760a3d8dec        47 minutes ago      2.55GB
```
