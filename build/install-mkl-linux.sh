#! /bin/bash





arch=$(uname -m)
echo $arch

cp -v build/linux-oneapi-yum-config /etc/yum.repos.d/oneAPI.repo
# We can expand this later to get the right libraries on other architectures if necessary
if [[ $arch =~ ([xX]86_64|[aA][mM][dD]64) ]]; then
    yum install -y intel-oneapi-mkl-devel
elif [[ $arch =~ ([xX]86|i386|i686) ]]; then
    # There is no intel-oneapi-mkl-devel package for 32bit operating systems.
#    yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
#    rpm --import https://yum.repose.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
    yum install -y intel-mkl
fi
