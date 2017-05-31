FROM nvidia/cuda:8.0-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install -y curl sudo aptitude bash-completion vim git
RUN apt-get install -y g++
RUN apt-get install -y libopencv-dev

ARG user=ian
ARG uid=1000
ARG gid=1000

RUN export uid=$uid gid=$gid user=$user && \
    mkdir -p /home/${user} && \
    echo "${user}:x:${uid}:${gid}:${user},,,:/home/${user}:/bin/bash" >> /etc/passwd && \
    echo "${user}:x:${uid}:" >> /etc/group && \
    echo "${user} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${user} && \
    chmod 0440 /etc/sudoers.d/${user} && \
    chown ${uid}:${gid} -R /home/${user}

ENV HOME /home/ian
ENV USER $user
ENV TERM xterm
USER $user

