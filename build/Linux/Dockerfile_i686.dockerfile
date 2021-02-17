FROM recombine_builder_manylinux_i686 AS manylinux2014_i686_esig

ENTRYPOINT ["linux32", "/bin/bash", "-c"]
SHELL ["linux32", "/bin/bash", "-c"]
RUN yum -y repolist
RUN yum -y install boost148-devel
RUN bash -c 'echo export BOOST_ROOT=/usr/include/boost148 >>~/.bashrc;\
echo export BOOST_LIB=/usr/lib/boost148 >> ~/.bashrc'
RUN bash -c 'cd /usr/lib;\
ln -s libboost_thread-mt.so.1.48.0 libboost_thread.so;\
ln -s libboost_system-mt.so.1.48.0 libboost_system.so'
CMD ["/bin/bash"]
