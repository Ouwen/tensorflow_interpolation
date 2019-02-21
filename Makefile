CXX := g++
PYTHON_BIN_PATH = python

SRCS = $(wildcard tensorflow_regular_interp/cc/kernels/*.cc) $(wildcard tensorflow_regular_interp/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

TARGET_LIB = tensorflow_regular_interp/python/ops/_regular_interp_ops.so


.PHONY: op
op: $(TARGET_LIB)

$(TARGET_LIB): $(SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

test: tensorflow_regular_interp/python/ops/regular_interp_ops_test.py tensorflow_regular_interp/python/ops/regular_interp_ops.py $(TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_regular_interp/python/ops/regular_interp_ops_test.py

pip_pkg: $(TARGET_LIB)
	./build_pip_pkg.sh make artifacts


.PHONY: clean
clean:
	rm -f $(TARGET_LIB)
