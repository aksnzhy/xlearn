#ifndef MEMALIGN
#define MEMALIGN

#ifdef _WIN32
#define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ?0 :errno)

#endif // _WIN32

#endif
