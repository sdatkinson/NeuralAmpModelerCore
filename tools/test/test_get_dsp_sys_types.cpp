// Regression test for issue #304.
//
// FreeBSD's <sys/types.h> and glibc's <sys/sysmacros.h> define major() and
// minor() as function-like macros, and unlike Apple's headers they keep them as
// macros in C++. Any consumer that includes one of those before NAM/get_dsp.h
// used to fail to compile, because nam::Version's member-initializer list read
// `major(major)` and the macro expanded.
//
// The macros are defined here rather than by including <sys/types.h> so the
// test reproduces the FreeBSD/glibc preprocessor state on every platform. They
// must go after get_dsp.h's dependency closure -- expanding them inside libc++'s
// own headers is a different, unrelated failure -- and are undefined at the end
// so they do not leak into the other tests included by run_tests.cpp.

#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/nam_file.h"

#define major(x) ((int)(((unsigned)(x) >> 24) & 0xff))
#define minor(x) ((int)((x) & 0xffffff))

#include "NAM/get_dsp.h"

#undef major
#undef minor

namespace test_get_dsp_sys_types
{
// Compiling this translation unit at all is the test; the assertions just keep
// the constructor from being optimized away and confirm the fields still land
// in the right order.
void test_version_builds_with_major_minor_macros_defined()
{
  const nam::Version version(1, 2, 3);
  assert(version.major == 1);
  assert(version.minor == 2);
  assert(version.patch == 3);
  assert(version.toString() == "1.2.3");
}
} // namespace test_get_dsp_sys_types
