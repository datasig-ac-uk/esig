FROM recombine_builder_manylinux_i686 AS manylinux2010_i686_esig

ENTRYPOINT ["/bin/bash", "-c"]
SHELL ["/bin/bash", "-c"]
RUN yum -y repolist
RUN yum -y install boost158-devel
RUN bash -c 'echo export BOOST_ROOT=/usr/include/boost148 >>~/.bashrc;\
echo export BOOST_LIB=/usr/lib/boost158 >> ~/.bashrc'
RUN bash -c 'cd /usr/lib;\
ln -s libboost_thread-mt.so.1.58.0 libboost_thread.so;\
ln -s libboost_system-mt.so.1.58.0 libboost_system.so'
CMD ["/bin/bash"]
