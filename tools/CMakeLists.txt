file(GLOB NAM_SOURCES_TOP "${CMAKE_CURRENT_SOURCE_DIR}/../NAM/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../NAM/*.c")
file(GLOB NAM_SOURCES_SUB "${CMAKE_CURRENT_SOURCE_DIR}/../NAM/*/*.cpp")
set(NAM_SOURCES ${NAM_SOURCES_TOP} ${NAM_SOURCES_SUB})

# TODO: add loadmodel and run_tests to TOOLS?
set(TOOLS benchmodel)

add_custom_target(tools ALL
	DEPENDS ${TOOLS} render)

set(AUDIO_DSP_TOOLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Dependencies/AudioDSPTools")
set(AUDIO_DSP_TOOLS_WAV_SOURCES "${AUDIO_DSP_TOOLS_DIR}/dsp/wav.cpp")

include_directories(tools ..)
include_directories(tools ${NAM_DEPS_PATH}/eigen)
include_directories(tools ${NAM_DEPS_PATH}/nlohmann)
include_directories(tools ${AUDIO_DSP_TOOLS_DIR}/dsp)

add_executable(loadmodel loadmodel.cpp ${NAM_SOURCES})
add_executable(benchmodel benchmodel.cpp ${NAM_SOURCES})
add_executable(bench_linear bench_linear.cpp ${NAM_SOURCES})
target_compile_features(bench_linear PUBLIC cxx_std_20)
set_target_properties(bench_linear PROPERTIES
	CXX_VISIBILITY_PRESET hidden
	INTERPROCEDURAL_OPTIMIZATION TRUE
	PREFIX ""
)
if (MSVC)
	target_compile_options(bench_linear PRIVATE "$<$<CONFIG:RELEASE>:/O2>")
else()
	target_compile_options(bench_linear PRIVATE "$<$<CONFIG:RELEASE>:-O3>")
endif()
add_executable(render render.cpp ${NAM_SOURCES} ${AUDIO_DSP_TOOLS_WAV_SOURCES})
target_compile_features(render PUBLIC cxx_std_20)
# AudioDSPTools wav.cpp has sign-compare issues; don't fail build
set_source_files_properties(${AUDIO_DSP_TOOLS_WAV_SOURCES} PROPERTIES COMPILE_FLAGS "-Wno-error")
set_target_properties(render PROPERTIES
	CXX_VISIBILITY_PRESET hidden
	INTERPROCEDURAL_OPTIMIZATION TRUE
	PREFIX ""
)
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	target_compile_definitions(render PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
if (MSVC)
	target_compile_options(render PRIVATE
		"$<$<CONFIG:DEBUG>:/W4>"
		"$<$<CONFIG:RELEASE>:/O2>"
	)
else()
	target_compile_options(render PRIVATE
		-Wall -Wextra -Wpedantic -Wstrict-aliasing -Wunreachable-code -Weffc++ -Wno-unused-parameter
		"$<$<CONFIG:DEBUG>:-Og;-ggdb;-Werror>"
		"$<$<CONFIG:RELEASE>:-Ofast>"
	)
endif()
add_executable(benchmodel_bufsize benchmodel_bufsize.cpp ${NAM_SOURCES})
add_executable(bench_a2_fast bench_a2_fast.cpp ${NAM_SOURCES})
target_compile_features(bench_a2_fast PUBLIC cxx_std_20)
set_target_properties(bench_a2_fast PROPERTIES
	CXX_VISIBILITY_PRESET hidden
	INTERPROCEDURAL_OPTIMIZATION TRUE
	PREFIX ""
)
if (MSVC)
	target_compile_options(bench_a2_fast PRIVATE
		"$<$<CONFIG:DEBUG>:/W4>"
		"$<$<CONFIG:RELEASE>:/O2>"
	)
else()
	target_compile_options(bench_a2_fast PRIVATE
		-Wall -Wextra -Wpedantic -Wstrict-aliasing -Wunreachable-code -Wno-unused-parameter
		"$<$<CONFIG:DEBUG>:-Og;-ggdb;-Werror>"
		"$<$<CONFIG:RELEASE>:-Ofast>"
	)
endif()
add_executable(bench_a2_planar bench_a2_planar.cpp ${NAM_SOURCES})
target_compile_features(bench_a2_planar PUBLIC cxx_std_20)
# No INTERPROCEDURAL_OPTIMIZATION here: link-time inlining of the whole library
# into the benchmark's own loop is not what a plugin build does, and measurably
# changes the result (about 3% on the 8-channel kernel on an M2).
set_target_properties(bench_a2_planar PROPERTIES
	CXX_VISIBILITY_PRESET hidden
	PREFIX ""
)
if (MSVC)
	target_compile_options(bench_a2_planar PRIVATE
		"$<$<CONFIG:DEBUG>:/W4>"
		"$<$<CONFIG:RELEASE>:/O2>"
	)
else()
	# -O3, not -Ofast. -ffast-math lets the compiler contract a multiply and an
	# add into an FMA across statement boundaries, which is exactly the freedom
	# this tool is checking has not been taken; measuring under it would make the
	# parity result meaningless.
	target_compile_options(bench_a2_planar PRIVATE
		-Wall -Wextra -Wpedantic -Wstrict-aliasing -Wunreachable-code -Wno-unused-parameter
		"$<$<CONFIG:DEBUG>:-Og;-ggdb;-Werror>"
		"$<$<CONFIG:RELEASE>:-O3>"
	)
endif()

# The A2 kernels are built optimised even for run_tests, which is otherwise -O0.
#
# test_a2_planar asserts memcmp equality against A2FastModel, and A2FastModel's
# 3-channel branch is a plain `a * b + c` chain whose bit-identity premise is
# that the compiler contracts it into an FMA. GCC only contracts in its
# optimisers, so at -O0 the reference computes something the planar kernels --
# which issue vfma unconditionally, being intrinsics -- cannot match, and the
# test fails on the very first sample. (Clang contracts during codegen, so it
# passes there either way; this is a GCC-only failure.) Every shipping build is
# optimised, so -O0 is the one configuration in which the reference is not the
# code the claim is about.
#
# Only these two translation units are raised, and only for this target, so
# every other test keeps the -O0 the allocation tracking was set up for. The
# allocation assertions that do reach these kernels all require *zero*
# allocations, and optimisation cannot introduce any.
set(A2_KERNEL_SOURCES "")
set(RUN_TESTS_NAM_SOURCES "")
foreach(_nam_src ${NAM_SOURCES})
	if(_nam_src MATCHES "/NAM/wavenet/a2_(fast|planar)\\.cpp$")
		list(APPEND A2_KERNEL_SOURCES "${_nam_src}")
	else()
		list(APPEND RUN_TESTS_NAM_SOURCES "${_nam_src}")
	endif()
endforeach()

add_library(run_tests_a2_kernels OBJECT ${A2_KERNEL_SOURCES})
target_compile_features(run_tests_a2_kernels PUBLIC cxx_std_20)
if(MSVC)
	target_compile_options(run_tests_a2_kernels PRIVATE /O2)
else()
	target_compile_options(run_tests_a2_kernels PRIVATE -O3)
endif()
# Keep assertions live here too, as they are in the rest of run_tests.
target_compile_options(run_tests_a2_kernels PRIVATE
	$<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:-UNDEBUG>
)

add_executable(run_tests run_tests.cpp test/allocation_tracking.cpp ${RUN_TESTS_NAM_SOURCES}
	$<TARGET_OBJECTS:run_tests_a2_kernels>)
# Compile run_tests without optimizations to ensure allocation tracking works correctly
# Also ensure assertions are enabled (NDEBUG is not defined) so tests actually run
set_target_properties(run_tests PROPERTIES COMPILE_OPTIONS "-O0")
# Ensure assertions are enabled for run_tests by removing NDEBUG if it was set
# Release/RelWithDebInfo/MinSizeRel build types automatically define NDEBUG
# We use a compile option to undefine it, which works on GCC, Clang, and MSVC
target_compile_options(run_tests PRIVATE
	$<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:-UNDEBUG>
)

# Option to enable ADDASSERT test (tests that assertions are actually enabled)
option(ENABLE_ASSERT_TEST "Enable ADDASSERT test to verify assertions are enabled" OFF)
if(ENABLE_ASSERT_TEST)
	target_compile_definitions(run_tests PRIVATE ADDASSERT)
endif()

source_group(NAM ${CMAKE_CURRENT_SOURCE_DIR} FILES ${NAM_SOURCES})

target_compile_features(${TOOLS} PUBLIC cxx_std_20)

set_target_properties(${TOOLS}
	PROPERTIES
	CXX_VISIBILITY_PRESET hidden
	INTERPROCEDURAL_OPTIMIZATION TRUE
	PREFIX ""
)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	target_compile_definitions(${TOOLS} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

if (MSVC)
	target_compile_options(${TOOLS} PRIVATE
		"$<$<CONFIG:DEBUG>:/W4>"
		"$<$<CONFIG:RELEASE>:/O2>"
	)
else()
	target_compile_options(${TOOLS} PRIVATE
		-Wall -Wextra -Wpedantic -Wstrict-aliasing -Wunreachable-code -Weffc++ -Wno-unused-parameter
		"$<$<CONFIG:DEBUG>:-Og;-ggdb;-Werror>"
		"$<$<CONFIG:RELEASE>:-Ofast>"
	)
endif()

# There's an error in eigen's
# /Users/steve/src/NeuralAmpModelerCore/Dependencies/eigen/Eigen/src/Core/products/GeneralBlockPanelKernel.h
# Don't let this break my build on debug:
set_source_files_properties(../NAM/dsp.cpp PROPERTIES COMPILE_FLAGS "-Wno-error")
set_source_files_properties(../NAM/conv1d.cpp PROPERTIES COMPILE_FLAGS "-Wno-error")
