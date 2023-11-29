#!/bin/bash -e

sudo echo ""

###########################################################
[ -e /var/run/docker.sock ] && \
        sudo chmod 777 /var/run/docker.sock

###########################################################
# docker_mount
###########################################################
DOCKER_MOUNT=/mnt/docker
[ ! -d ${DOCKER_MOUNT} ] && (sudo mkdir -p ${DOCKER_MOUNT} && sudo chmod 7777 ${DOCKER_MOUNT})

REPOSITORY=pytorch/pytorch
docker rmi -f $(docker images -a | grep "^${REPOSITORY}" | awk '{print $3}') || true

###########################################################
# docker run
###########################################################
docker build -t="${REPOSITORY}:vim" .
docker run --rm -it -v ${DOCKER_MOUNT}:/docker_mount --name mmdocker \
	--shm-size=8gb \
	--gpus all --runtime=nvidia \
	${REPOSITORY}:vim


exit 0
