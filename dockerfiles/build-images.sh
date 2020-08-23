
#!/bin/bash

set -e

docker build -t spark-base:2.3.1 ./base
docker build -t spark-master:2.3.1 ./spark-master
docker build -t spark-worker:2.3.1 ./spark-worker
