FROM quay.io/pypa/manylinux2014_i686:latest

SHELL ["/bin/bash", "-c"]

RUN yum -y install boost-thread boost-system boost-devel
RUN curl -sSL -o openblas.tar.gz https://anaconda.org/multibuild-wheels-staging/openblas-libs/v0.3.19-22-g5188aede/download/openblas-v0.3.19-22-g5188aede-manylinux2010_i686.tar.gz
RUN tar -xzf openblas.tar.gz

COPY . /esig
WORKDIR /esig

ENTRYPOINT ["/bin/bash"]