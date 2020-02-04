#!/bin/bash
# Top-level build script.

for arch in i686 # x86_64 installs fail with gcc internal compiler error
do 
   docker run --rm -v ${PWD}:/data esig_builder_linux_${arch} "source ~/.bashrc; cd /data; source linux_wheel_maker.sh $arch"
done
