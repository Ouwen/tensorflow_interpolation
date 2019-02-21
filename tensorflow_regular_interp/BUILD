licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = 'python/ops/_regular_interp_ops.so',
    srcs = [
        "cc/kernels/regular_interp_kernels.cc",
        "cc/ops/regular_interp_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"]
)

py_library(
    name = "regular_interp_ops_py",
    srcs = ([
        "python/ops/regular_interp_ops.py",
    ]),
    data = [
        ":python/ops/_regular_interp_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "regular_interp_ops_py_test",
    srcs = [
        "python/ops/regular_interp_ops_test.py"
    ],
    main = "python/ops/regular_interp_ops_test.py",
    deps = [
        ":regular_interp_ops_py",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "regular_interp_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":regular_interp_ops_py"
    ],
    srcs_version = "PY2AND3",
)
