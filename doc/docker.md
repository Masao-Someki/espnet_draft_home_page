# Docker

## Execute in docker
To work inside a docker container, execute `run.sh` located inside the docker directory.
It will download the requested image and build a container to execute the main program specified by the following GPU, ASR example, and outside directory information, as follows:

```sh
$ cd docker
$ ./run.sh --docker-gpu 0 --docker-egs chime4/asr1 --docker-folders /export/corpora4/CHiME4/CHiME3 --dlayers 1 --ngpu 1
```

Optionally, you can set the CUDA version with the arguments `--docker-cuda` respectively (default version set at CUDA=9.1). The docker container can be built based on the CUDA installed in your computer if you empty this arguments.
By default, all GPU-based images are built with NCCL v2 and CUDNN v7.
The arguments required for the docker configuration have a prefix "--docker" (e.g., `--docker-gpu`, `--docker-egs`, `--docker-folders`). `run.sh` accept all normal ESPnet arguments, which must be followed by these docker arguments.
All docker containers are executed using the same user as your login account. If you want to run the docker in root access, add the flag `--is-root` to command line. In addition, you can pass any environment variable using `--docker-env` (e.g., `--docker-env "foo=path"`)

## ESPnet 2 Recipes

To work with recipes of ESPnet 2, you will need to add the flag `--is-egs2` to the command line:

```sh
$ cd docker
$ ./run.sh --docker-gpu 0 --is-egs2 --docker-egs an4/asr1 --ngpu1
```

Remember to add the flag before the arguments you want to pass to the recipe `run.sh` file.

## Using GPU-based containers

You can run any bash script implemented in the egs folder using `--docker-cmd`:

```sh
$ cd docker
$ ./run.sh --docker-gpu 0 --docker-egs chime4/asr1 --docker-cmd foo.sh --arg_1 <arg_1> --arg_2 <arg_2>
```

The arguments for the desired script should follow the docker arguments. `run.sh` is the default script to be executed.

Multiple GPUs should be specified with the following options:

```sh
$ cd docker
$ ./run.sh --docker-gpu 0,1,2 --docker-egs chime5/asr1 --docker-folders /export/corpora4/CHiME5 --ngpu 3
```

Note that all experimental files and results are created under the normal example directories (`egs/<example>/`).

Multiple folders and environment variables should be specified with commas and without spaces:

```sh
$ cd docker
$ ./run.sh --docker-gpu 0 --docker-egs chime4/asr1 --docker-folders /export/corpus/CHiME4,/export/corpus/LDC/LDC93S6B,/export/corpus/LDC/LDC94S13B --docker-env "CHIME4_CORPUS=/export/corpus/CHiME4/CHiME3,WSJ0_CORPUS=/export/corpus/LDC/LDC93S6B,WSJ1_CORPUS=/export/corpus/LDC/LDC94S13B" --ngpu 1
```

Remember that for some recipes, you first need to download the Corpus before running the experiments, such as CHiME, WSJ, and LDC corporas. You will need to set the directories where these were downloaded and replace them in the recipe (e.g.: `CHIME4_CORPUS=/<dir_where_chime4_was_downloaded>/CHiME4/CHiME3`)

## Using CPU-based container

You can train a model in CPU using the following command:

```sh
$ cd docker
$ ./run.sh --docker-gpu -1 --docker-egs an4/asr1 --ngpu 0
```

The script will build a docker if your are using a `user` different from `root` user. To use containers with `root` access
add the flag `--is-root` to the command line.


## Local builds

When building the docker container on a local machine, the espnet source is downloaded from the github espnet master branch.
However, in some cases, "local" builds are preferable, that are built based on the source code from the local repository:

1. After writing own modifications on the espnet code, the build environment, etc., and to test it in the docker container. Prebuilt docker containers do not import these.

2. Reproducability: It is possible to go back to an espnet version at a certain commit and test the neural network with an older version of a library.

The script `build.sh` supports making local builds for this purpose. During the docker build process, the local espnet source code is imported through a git archive based on git HEAD (the previous commit), and copied over within a file.

For example, a local build that the base image from Docker Hub (`espnet/espnet:runtime`, based on Ubuntu 16), that already contains a kaldi installation, using Cuda 10.0:

```
./build.sh local 10.0
```

Also, docker images can also be built based on the Ubuntu version specified in `prebuilt/runtime/Dockerfile` (currently set to Ubuntu 18.04), in this example case using the cpu:

```
./build.sh fully_local cpu
```

Local container builds then are started by adding the flag `--is-local` when using `run.sh`, e.g., for the Cuda 10.0 image:

```
$ ./run.sh --is-local --docker_cuda 10.0 --docker_gpu 0 ...
```


## Deprecated

Containers build on ubuntu-16.04 will be deprecated and no longer receive support. However, these container will remain in Docker Hub.
To use containers with ubuntu 16.04, empty the flag `--docker_os`.

## Tags

- Runtime: Base image for ESPnet. It includes libraries and Kaldi installation.
- CPU: Image to execute only in CPU.
- GPU: Image to execute examples with GPU support.

## Ubuntu 18.04

Pytorch 1.3.1, No warp-ctc:

- [`cuda10.1-cudnn7` (*docker/prebuilt/gpu/10.1/cudnn7/Dockerfile*)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/devel/gpu/10.1/cudnn7/Dockerfile)

Pytorch 1.0.1, warp-ctc:

- [`cuda10.0-cudnn7` (*docker/prebuilt/gpu/10.0/cudnn7/Dockerfile*)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/devel/gpu/10.0/cudnn7/Dockerfile)
- [`cpu-u18` (*docker/prebuilt/devel/Dockerfile*)](https://github.com/espnet/espnet/tree/master/docker/prebuilt/devel/Dockerfile)
