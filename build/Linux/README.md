### Prerequisites: Docker

This process relies on having [Docker](https://docs.docker.com/)
installed. There are minor differences in the Docker commands to
run if you are using Windows (and Docker for Windows) or a POSIX system, but
the Dockerfiles themselves are identical.

### To build on Linux or OSX.

Run the literate script `build-wheels.sh`.

### To build on Windows

There is no top-level Windows script to build the Linux wheels. Instead, manually carry out the appropriate steps from `build-wheels.sh`, and then run the following. Ensure that Docker for Windows has "experimental features" enabled so that it can run linux containers.
```
docker run --platform=linux --rm -v "%CD%":/data esig_builder_linux_i686 "source ~/.bashrc ; cd /data; source linux_wheel_maker.sh i686; done;"
docker run --platform=linux --rm -v "%CD%":/data esig_builder_linux_x86_64 "source ~/.bashrc ; cd /data; source linux_wheel_maker.sh x86_64; done;"
```
