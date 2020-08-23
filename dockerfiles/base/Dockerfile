#FROM python:3.6.8-alpine
#FROM python:3.6.8-alpine3.8 as BASE
#FROM java:8-jdk-alpine
FROM tensorflow/tensorflow:nightly-py3 as BASE
#FROM ubuntu:14.04

ENV DAEMON_RUN=true
ENV SPARK_VERSION=2.4.5
ENV HADOOP_VERSION=2.7
ENV SCALA_VERSION=2.12.4
ENV SCALA_HOME=/usr/share/scala

#RUN apk update
#RUN apk add make automake gcc g++ subversion

#RUN apt-get install gcc g++ 
#RUN pip3 install numpy
#RUN apk add --no-cache openjdk8-jre
RUN apt-get update && apt-get -y install software-properties-common 
RUN add-apt-repository ppa:openjdk-r/ppa && apt-get update && apt-get -y install openjdk-8-jre

#RUN apk add --no-cache --virtual=.build-dependencies wget ca-certificates && \
#    apk add --no-cache bash curl jq && \
#    cd "/tmp" && \
#    wget --no-verbose "https://downloads.typesafe.com/scala/${SCALA_VERSION}/scala-${SCALA_VERSION}.tgz" && \
#    tar xzf "scala-${SCALA_VERSION}.tgz" && \
#    mkdir "${SCALA_HOME}" && \
#    rm "/tmp/scala-${SCALA_VERSION}/bin/"*.bat && \
#    mv "/tmp/scala-${SCALA_VERSION}/bin" "/tmp/scala-${SCALA_VERSION}/lib" "${SCALA_HOME}" && \
#    ln -s "${SCALA_HOME}/bin/"* "/usr/bin/" && \
#    apk del .build-dependencies && \
#    rm -rf "/tmp/"*


RUN apt-get install  wget ca-certificates && \
    apt-get install  bash curl jq && \
    cd "/tmp" && \
    wget --no-verbose "https://downloads.typesafe.com/scala/${SCALA_VERSION}/scala-${SCALA_VERSION}.tgz" && \
    tar xzf "scala-${SCALA_VERSION}.tgz" && \
    mkdir "${SCALA_HOME}" && \
    rm "/tmp/scala-${SCALA_VERSION}/bin/"*.bat && \
    mv "/tmp/scala-${SCALA_VERSION}/bin" "/tmp/scala-${SCALA_VERSION}/lib" "${SCALA_HOME}" && \
    ln -s "${SCALA_HOME}/bin/"* "/usr/bin/" && \
    #iapk del .build-dependencies && \
    rm -rf "/tmp/"*


#RUN apk update
#RUN apk add make automake gcc g++ subversion  

#RUN apk add --update py3-pip
#RUN pip install numpy

RUN apt-get install python3-pip

#Scala instalation
#RUN export PATH="/usr/local/sbt/bin:$PATH" &&  apk update && apk add ca-certificates wget tar && mkdir -p "/usr/local/sbt" && wget -qO - --no-check-certificate "https://cocl.us/#sbt-0.13.16.tgz" | tar xz -C /usr/local/sbt --strip-components=1 && sbt sbtVersion

#RUN  apk add --update --no-cache python3.6

#RUN apt-get -y install python3.6 && pip install tensorflow==1.15.1

#RUN apk add --no-cache  python3.6 python3.6-dev python3-pip python3.6-venv
RUN wget http://apache.mirror.anlx.net/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz

RUN tar -xzf spark-2.4.5-bin-hadoop2.7.tgz && \
    mv spark-2.4.5-bin-hadoop2.7 /spark && \
    rm spark-2.4.5-bin-hadoop2.7.tgz
RUN apt-get -y install python3.6 vim git && pip install tensorflow==1.15.0 && apt-get install software-properties-common && apt-get update 
