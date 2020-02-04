for arch in x86_64 # i686
do 
   docker run --rm -v ${PWD}:/data esig_builder_linux_${arch} <<-BLOCK
      source ~/.bashrc
      cd /data
      source linux_wheel_maker.sh $arch
   BLOCK
done
