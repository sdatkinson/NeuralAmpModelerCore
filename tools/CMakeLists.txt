file(GLOB_RECURSE NAM_SOURCES ../NAM/*.cpp ../NAM/*.c ../NAM*.h)

# TODO: add loadmodel and run_tests to TOOLS?
set(TOOLS benchmodel)

add_custom_target(tools ALL
	DEPENDS ${TOOLS})

include_directories(tools ..)
include_directories(tools ${NAM_DEPS_PATH}/eigen)
include_directories(tools ${NAM_DEPS_PATH}/nlohmann)

add_executable(loadmodel loadmodel.cpp ${NAM_SOURCES})
add_executable(benchmodel benchmodel.cpp ${NAM_SOURCES})
add_executable(memory_usage memory_usage.cpp)
add_executable(run_tests run_tests.cpp test/allocation_tracking.cpp ${NAM_SOURCES})
# Compile run_tests without optimizations to ensure allocation tracking works correctly
# Also ensure assertions are enabled (NDEBUG is not defined) so tests actually run
set_target_properties(run_tests PROPERTIES COMPILE_OPTIONS "-O0")
# Benchmodel should be built with NAM_PROFILING set
target_compile_definitions(benchmodel PRIVATE NAM_PROFILING)
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
target_compile_features(memory_usage PUBLIC cxx_std_20)

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
