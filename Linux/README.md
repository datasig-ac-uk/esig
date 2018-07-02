### Building esig for Linux

This process relies on having [Docker](https://docs.docker.com/)
installed and running.  There are minor differences in the docker commands to
run if you are using Windows (and docker-for-windows) or a posix system, but
the Dockerfiles themselves are identical.

Two python wheels will be created, one for 64-bit (x86_64)
and one for 32-bit (i686).
