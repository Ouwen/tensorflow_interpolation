# TensorFlow Regular Interp

## Build regular_interp Op

### Setup Docker Container
You are going to build the op inside a Docker container. Pull the provided Docker image from TensorFlow's Docker hub and start a container.

```bash
  docker pull tensorflow/tensorflow:custom-op
  docker run -it tensorflow/tensorflow:custom-op /bin/bash
```

Inside the Docker container, clone this repository. The code in this repository came from the [Adding an op](https://www.tensorflow.org/extend/adding_an_op) guide.
```bash
git clone https://github.com/tensorflow/custom-op.git
cd custom-op
```

### Build PIP Package
You can build the pip package with either Bazel or make.

With bazel:
```bash
  ./configure.sh
  bazel build build_pip_pkg
  bazel-bin/build_pip_pkg artifacts
```

With Makefile:
```bash
  make pip_pkg
```

### Install and Test PIP Package
Once the pip package has been built, you can install it with,
```bash
pip install artifacts/*.whl
```

### Acknowledgements

Thanks to the Tensorflow team for creating the example-op guide.
Thanks to @dfm for the interp kernel logic.
