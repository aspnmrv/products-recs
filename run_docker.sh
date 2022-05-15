echo killing old docker processes
docker-compose rm -fs

echo building docker containers
export DOCKER_REGISTRY=local
docker-compose up --build