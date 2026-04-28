cmake_minimum_required(VERSION 3.10)

# Make sure this matches ./NAM/version.h!
project(NAM VERSION 0.4.0)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use libc++ on macOS, system default on Linux
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

option(NAM_USE_LIBCXX_LINUX "Use libc++ on Linux (Clang + libc++-dev packages; tests SlimmableWavenet _LIBCPP_VERSION path)" OFF)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	include_directories(SYSTEM /usr/local/include)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	if (NAM_USE_LIBCXX_LINUX)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
		# Clang links libc++/libc++abi; do not pull in libstdc++fs (libstdc++).
	else()
		link_libraries(stdc++fs)
	endif()
elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
else()
	message(FATAL_ERROR "Unrecognized Platform!")
endif()

set(NAM_DEPS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Dependencies")
include_directories(SYSTEM "${NAM_DEPS_PATH}/eigen")

# Build the specialized A2 fast-path WaveNet (A2 standard + A2 nano). When ON,
# models whose config matches the A2 shape signature are routed to a hand-optimized
# WaveNet implementation instead of the generic one. When OFF, all models go
# through the generic path.
option(NAM_ENABLE_A2_FAST "Build the A2 fast-path WaveNet" ON)
if(NAM_ENABLE_A2_FAST)
	add_compile_definitions(NAM_ENABLE_A2_FAST)
endif()

add_subdirectory(tools)

#file(MAKE_DIRECTORY build/tools)

#add_custom_target(copy_tools ALL
#	${CMAKE_COMMAND} -E copy "$<TARGET_FILE:tools>" tools/
#	DEPENDS tools
#)
