#! /bin/bash

cp -v build/linux-oneapi-yum-config /etc/yum.repos.d/oneAPI.repo


arch=$(uname -m)

# We can expand this later to get the right libraries on other architectures if necessary
if [[ $arch =~ ([xX]86_64|[aA][mM][dD]64) ]]; then
    yum install intel-oneapi-mkl-devel
elif [[ $arch =~ ([xX]86|i386|i686) ]]; then
    yum install intel-oneapi-mkl-devel
fi