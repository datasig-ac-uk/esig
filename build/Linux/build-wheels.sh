#!/bin/bash
# Top-level build script.

for arch in i686 # x86_64 fails with gcc internal compiler error running esig installer
do 
   docker run --rm -v ${PWD}:/data esig_builder_linux_${arch} "source ~/.bashrc; cd /data; source linux_wheel_maker.sh $arch"
done
