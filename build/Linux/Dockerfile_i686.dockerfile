FROM quay.io/pypa/manylinux1_i686:2020-01-31-d8fa357 AS manylinux1_i686_boost
## from the folder with this file and the context execute
## docker build -t manylinux1_i686_boost -f Dockerfile.dockerfile .
ENTRYPOINT ["linux32", "/bin/bash", "-c"]
SHELL ["linux32", "/bin/bash", "-c"]
RUN yum -y repolist
RUN yum -y install boost148-devel
RUN bash -c 'echo export BOOST_ROOT=/usr/include/boost148 >>~/.bashrc;\
echo export BOOST_LIB=/usr/lib/boost148 >> ~/.bashrc'
RUN bash -c 'cd /usr/lib;\
ln -s libboost_thread-mt.so.1.48.0 libboost_thread.so;\
ln -s libboost_system-mt.so.1.48.0 libboost_system.so'
# Preamble required to install recombine
RUN yum install -y wget
CMD ["/bin/bash"]
