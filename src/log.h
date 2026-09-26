#ifndef MIE_INTERNAL_LOG_H_
#define MIE_INTERNAL_LOG_H_

#include <cstdio>

#define MIE_LOGW(...)                                    \
  do {                                                   \
    std::fprintf(stderr, "[mie][W] " __VA_ARGS__);       \
    std::fprintf(stderr, "\n");                          \
  } while (0)

#endif  // MIE_INTERNAL_LOG_H_
