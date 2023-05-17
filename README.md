# popcorn

A set of kernels exploded with manual unrolling, with a common
parallelism:

- outer parallelism shards work to cores, either whole network or chunk of nodes
- inner parallelism shards batch to hide memory latency, optinally w/ SIMD

This generally works for GPUs, and it quite OK for CPUs as long as
the working set stays in L3 cache (vs fused kernels).  The benefit is
kernel reusability.

ISPC & OpenCL aren't the hot parallel APIs these days but they do cover
available hardware (CPU, m1 GPU, Intel GPU, NVidia & AMD) with a
simple programming model.

# TODO

- add bold filter
- ispc versions of opencl kernels
- ispc pcg or philox
- add & defuse mpr delay
