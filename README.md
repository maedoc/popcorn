# popcorn

A set of kernels exploded with manual unrolling, with a common
parallelism:

- outer parallelism shards work to cores, either whole network or chunk of nodes
- inner parallelism shards batch to hide memory latency, optinally w/ SIMD

This generally works for GPUs, and it quite OK for CPUs as long as
the working set stays in L3 cache (vs fused kernels).  The benefit is
kernel reusability.


# TODO

- add bold filter
- ispc versions of opencl kernels
- ispc pcg or philox
- add & defuse mpr delay
