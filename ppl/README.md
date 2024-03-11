# Kernel Implementations

This folder provides two static librarys `ppl-openmp` and `ppl-hybrid`. If on CPU-only application, use the first one; on GPU or CPU/GPU use the second one. This is because the underlying memory allocation uses `cudaMallocManaged` in the hybrid approach, whereas `ppl-openmp` uses `new[]`. Do not link to both of them.  

## Special Notes

For hybrid or GPU-only approach, will need to attach the memories to a stream before using them.
