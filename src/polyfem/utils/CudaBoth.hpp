#pragma once

#ifdef __CUDACC__
#define POLYFEM_BOTH __host__ __device__
#else
#define POLYFEM_BOTH
#endif
