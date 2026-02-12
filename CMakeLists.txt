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

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	include_directories(SYSTEM /usr/local/include)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	link_libraries(stdc++fs)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
else()
	message(FATAL_ERROR "Unrecognized Platform!")
endif()

set(NAM_DEPS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Dependencies")

set(NAM_CORE_SOURCES
	NAM/activations.cpp
	NAM/conv1d.cpp
	NAM/convnet.cpp
	NAM/dsp.cpp
	NAM/get_dsp.cpp
	NAM/lstm.cpp
	NAM/ring_buffer.cpp
	NAM/util.cpp
	NAM/wavenet.cpp
	NAM/nam_c_api.cpp
)

add_library(nam_static STATIC ${NAM_CORE_SOURCES})
add_library(nam_shared SHARED ${NAM_CORE_SOURCES})
add_library(nam::static ALIAS nam_static)
add_library(nam::shared ALIAS nam_shared)

set(NAM_LIBRARY_TARGETS nam_static nam_shared)
foreach(target ${NAM_LIBRARY_TARGETS})
	target_compile_features(${target} PUBLIC cxx_std_20)
	target_include_directories(${target}
		PUBLIC
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
			$<INSTALL_INTERFACE:include>
		PRIVATE
			${NAM_DEPS_PATH}/eigen
			${NAM_DEPS_PATH}/nlohmann
	)
	set_target_properties(${target}
		PROPERTIES
		CXX_VISIBILITY_PRESET hidden
		VISIBILITY_INLINES_HIDDEN YES
		POSITION_INDEPENDENT_CODE ON
	)
	if (MSVC)
		target_compile_options(${target} PRIVATE
			"$<$<CONFIG:DEBUG>:/W4>"
			"$<$<CONFIG:RELEASE>:/O2>"
		)
	else()
		target_compile_options(${target} PRIVATE
			-Wall -Wextra -Wpedantic -Wstrict-aliasing -Wunreachable-code -Weffc++
			-Wno-unused-parameter
			"$<$<CONFIG:DEBUG>:-Og;-ggdb;-Werror>"
			"$<$<CONFIG:RELEASE>:-Ofast>"
		)
	endif()
endforeach()

target_compile_definitions(nam_shared PRIVATE NAM_BUILD_SHARED)
set_target_properties(nam_static PROPERTIES OUTPUT_NAME nam)
set_target_properties(nam_shared PROPERTIES
	OUTPUT_NAME nam
	VERSION ${PROJECT_VERSION}
	SOVERSION ${PROJECT_VERSION_MAJOR}
)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	set_target_properties(nam_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
	target_compile_definitions(nam_static PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
	target_compile_definitions(nam_shared PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# There's an error in eigen's GeneralBlockPanelKernel.h in some debug builds.
set_source_files_properties(NAM/dsp.cpp PROPERTIES COMPILE_FLAGS "-Wno-error")
set_source_files_properties(NAM/conv1d.cpp PROPERTIES COMPILE_FLAGS "-Wno-error")

install(TARGETS nam_static nam_shared EXPORT NAMTargets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
)
install(FILES NAM/nam_c_api.h DESTINATION include/NAM)
install(EXPORT NAMTargets
	FILE NAMTargets.cmake
	NAMESPACE nam::
	DESTINATION lib/cmake/NAM
)

add_subdirectory(tools)

#file(MAKE_DIRECTORY build/tools)

#add_custom_target(copy_tools ALL
#	${CMAKE_COMMAND} -E copy "$<TARGET_FILE:tools>" tools/
#	DEPENDS tools
#)
