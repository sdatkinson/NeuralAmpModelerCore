#pragma once

#if defined(_MSC_VER) && !defined(__llvm__)
  #define NAM_RESTRICT __restrict
#else
  #define NAM_RESTRICT __restrict__
#endif
