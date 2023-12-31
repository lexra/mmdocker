#!/bin/bash -e

sudo echo ""

###########################################################
[ -e /var/run/docker.sock ] && \
        sudo chmod 777 /var/run/docker.sock

###########################################################
# docker_mount
###########################################################
DOCKER_MOUNT=/work/docker_mount
[ ! -d ${DOCKER_MOUNT} ] && (sudo mkdir -p ${DOCKER_MOUNT} && sudo chmod 7777 ${DOCKER_MOUNT})

TAG=mmdocker
REPOSITORY=lexra/${TAG}
docker rmi -f $(docker images -a | grep "^${REPOSITORY}" | awk '{print $3}') || true

###########################################################
# docker run
###########################################################
docker build -t="${REPOSITORY}" .
docker run --rm -it -v ${DOCKER_MOUNT}:/docker_mount --name mmdocker \
	--shm-size=4gb \
	--gpus all --runtime=nvidia \
	${REPOSITORY}

exit 0
