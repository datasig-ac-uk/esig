#! /bin/bash


arch=$(uname -m)
echo $arch
yum install -y curl zip unzip tar
bash ./build/vcpkg/bootstrap-vcpkg.sh

build/vcpkg/vcpkg install boost-system boost-thread boost-container boost-multiprecision

#yum install -y boost-devel


# We can expand this later to get the right libraries on other architectures if necessary
if [[ $arch =~ ([xX]86_64|[aA][mM][dD]64) ]]; then
    cp -v build/linux-oneapi-yum-config /etc/yum.repos.d/oneAPI.repo
    yum install -y intel-oneapi-mkl-devel
elif [[ $arch =~ ([xX]86|i386|i686) ]]; then
    # There is no intel-oneapi-mkl-devel package for 32bit operating systems.
#    yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
#    rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB

#    yum install -y --skip-broken intel-oneapi-common-vars intel-oneapi-openmp-common intel-oneapi-openmp-32bit intel-oneapi-tbb-32bit intel-oneapi-tbb-common intel-oneapi-mkl-common intel-oneapi-mkl-32bit

    ## Borrowing some code to import openblas from Numpy for getting openblas from anaconda's repository
    openblas_long="v0.3.19-22-g5188aede"
    base_loc="https://anaconda.org/multibuild-wheels-staging/openblas-libs/${openblas_long}/download"
    if [[ $AUDITWHEEL_POLICY ]]; then
      echo "Auditwheel policy $AUDITWHEEL_POLICY and arch ${AUDITWHEEL_ARCH}"
      url="${base_loc}/openblas-${openblas_long}-${AUDITWHEEL_POLICY}_${AUDITWHEEL_ARCH}.tar.gz"
    else
      url="${base_loc}/linux_${arch}"
    fi

    pushd / || exit 1
    if curl -SL "${url}" | tar -xvzf -; then
      echo "Downloaded $?"
    else
      ## download openblas source and compile?
      echo "Cannot install openblas here yet"
      exit 1
    fi
    popd || exit 1


fi
