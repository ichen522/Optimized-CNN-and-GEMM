load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "1",
    srcs = ["non_coding_q1.cc", "q.h"],
    copts = [
        "-O3",
        "-march=native",
    ],
    visibility = ["//visibility:public"],
)