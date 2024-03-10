#Shared Code Between CPU and GPU

There are some functions that can be used for both CPU and GPU. They process on input data, and is procedural. Can describe the algorithm. They are executed the same way.

* `morton.h` functions that converts `xyz` positions (glm::vec4) to 32-bit `unsigned int` morton code.
* `oct_v2.h` functions that construct the octree, given radix tree and other things
* `edge_count.h`

Where as kernels like

* `sort` 
* `prefix sum`
* `unique`

should have different implementations. E.g., GPU want to use Radix Sort. 

## However

the Binary Radix tree kernel actual uses the same algorithm, but they depends on specifc `clz` count leading zeros function, which is different implementation on CPU and on GPU.
