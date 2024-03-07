#pragma once

#include "handlers/pipe.cuh"
#include "kernels_fwd.h"

namespace gpu::v2 {

void dispatch_BuildOctree(int grid_size, cudaStream_t stream, const Pipe& pipe);

}  // namespace gpu::v2
