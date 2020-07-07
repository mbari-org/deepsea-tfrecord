FROM tensorflow/tensorflow:1.13.0rc0-py3 as builder

RUN apt-get update && apt-get install -y wget && \
    apt-get install -y git && \
    apt-get install -y unzip && \
    apt-get install -y libglib2.0-0

# Install needed proto binary and clean
WORKDIR /tmp/protoc3
RUN wget https://github.com/google/protobuf/releases/download/v3.4.0/protoc-3.4.0-linux-x86_64.zip
RUN unzip /tmp/protoc3/protoc-3.4.0-linux-x86_64.zip
RUN mv /tmp/protoc3/bin/* /usr/local/bin/
RUN mv /tmp/protoc3/include/* /usr/local/include/
RUN rm -Rf /tmp/protoc3

ENV APP_HOME /app
WORKDIR /app
ENV PATH=${PATH}:/app

# Install Tensorflow models to get some helper functions for creating records
RUN git clone https://github.com/tensorflow/models.git tensorflow_models && cd tensorflow_models &&  git checkout tags/v1.13.0
ENV PYTHONPATH=/app:/app/tensorflow_models/research:/app/tensorflow_models/research/slim:/app/tensorflow_models/research/object_detection

# build protobufs
RUN cd /app/tensorflow_models/research && protoc object_detection/protos/*.proto --python_out=.

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install Cython==0.28.2
RUN python3 -m pip install pycocotools==2.0.0
COPY src/main/ /app
COPY requirements.txt /app
RUN pip install -r /app/requirements.txt
ARG DOCKER_GID
ARG DOCKER_UID

# Add non-root user and fix permissions
RUN groupadd --gid $DOCKER_GID docker && adduser --uid $DOCKER_UID --gid $DOCKER_GID --disabled-password --quiet --gecos "" docker_user
RUN chown -Rf docker_user:docker /app

USER docker_user
WORKDIR /app
RUN chown -Rf docker_user:docker /app
ENTRYPOINT ["python3", "/app/create_tfrecord.py"]

# Add test, building on
FROM builder as testrunner
WORKDIR /
USER root
RUN python3 -m pip install nose==1.3.7
COPY src/test/ /test
COPY data /test
RUN nosetests /test/test.py || true
