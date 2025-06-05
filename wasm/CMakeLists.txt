# Find all NAM (Neural Amp Modeler) source files recursively
file(GLOB_RECURSE NAM_SOURCES ../NAM/*.cpp ../NAM/*.c ../NAM*.h)

# Define the tools we want to build
set(TOOLS t3k-wasm-module)

# Create a custom target that depends on all tools
add_custom_target(tools ALL
	DEPENDS ${TOOLS})

# Include necessary directories for compilation
include_directories(tools ..)
include_directories(tools ${NAM_DEPS_PATH}/eigen)  # Eigen library for matrix operations
include_directories(tools ${NAM_DEPS_PATH}/nlohmann)  # JSON library

# Create the t3k-wasm-module executable
add_executable(t3k-wasm-module t3k-wasm-module.cpp ${NAM_SOURCES})

# Define compilation flags for Web Audio compatibility
# Web Audio API only supports 32-bit float processing
add_definitions(-DNAM_SAMPLE_FLOAT)
# Disable Eigen's stack allocation limit to prevent stack overflow
add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=0)

# Group NAM source files in the IDE
source_group(NAM ${CMAKE_CURRENT_SOURCE_DIR} FILES ${NAM_SOURCES})

# Set C++17 as the required standard
target_compile_features(${TOOLS} PUBLIC cxx_std_17)

# Configure build properties for all tools
set_target_properties(${TOOLS}
	PROPERTIES
	CXX_VISIBILITY_PRESET hidden  # Hide symbols by default
	INTERPROCEDURAL_OPTIMIZATION TRUE  # Enable link-time optimization
	PREFIX ""  # Remove the 'lib' prefix from output files
)

# Set specific compilation flags for t3k-wasm-module
set_target_properties(t3k-wasm-module PROPERTIES COMPILE_FLAGS "-Os -pthread")

# Configure Emscripten-specific linker flags for WebAssembly compilation
# These flags enable:
# - Web Audio Worklet support
# - Web Workers
# - Memory growth
# - Async operations
# - Threading support
set_target_properties(t3k-wasm-module PROPERTIES LINK_FLAGS "-Os -sEXPORTED_FUNCTIONS=stringToUTF8,_malloc,_setDsp,_free -sENVIRONMENT=web,worker -sEXPORTED_RUNTIME_METHODS=ccall -sAUDIO_WORKLET -sWASM_WORKERS -sALLOW_MEMORY_GROWTH -sINITIAL_MEMORY=1024MB -sTOTAL_STACK=512MB -sASYNCIFY -pthread")

# TODO: Support JS Modules
# set_target_properties(t3k-wasm-module PROPERTIES LINK_FLAGS "-Os -sEXPORTED_FUNCTIONS=stringToUTF8,_malloc,_setDsp,_free -sENVIRONMENT=web,worker -sEXPORTED_RUNTIME_METHODS=ccall -sAUDIO_WORKLET -sWASM_WORKERS -sALLOW_MEMORY_GROWTH -sINITIAL_MEMORY=1024MB -sTOTAL_STACK=512MB -sASYNCIFY -sMODULARIZE -sASSERTIONS -sWEBAUDIO_DEBUG -sEXPORT_NAME=coreModule -sEXPORT_ES6=1")

# Windows-specific definitions
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	target_compile_definitions(${TOOLS} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Set compiler-specific flags
if (MSVC)
	# Microsoft Visual C++ specific flags
	target_compile_options(${TOOLS} PRIVATE
		"$<$<CONFIG:DEBUG>:/W4>"  # Warning level 4 in debug mode
		"$<$<CONFIG:RELEASE>:/O2>"  # Optimization level 2 in release mode
	)
else()
	# GCC/Clang specific flags
	target_compile_options(${TOOLS} PRIVATE
		-Wall -Wextra -Wpedantic -Wstrict-aliasing -Wunreachable-code -Weffc++ -Wno-unused-parameter
		"$<$<CONFIG:DEBUG>:-Og;-ggdb;-Werror>"  # Debug flags with warnings as errors
		"$<$<CONFIG:RELEASE>:-Os>"  # Size optimization in release mode
	)
endif()