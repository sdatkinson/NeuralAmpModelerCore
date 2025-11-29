file(GLOB_RECURSE NAM_SOURCES ../NAM/*.cpp ../NAM/*.c ../NAM*.h)
file(GLOB TEST_SOURCES test/*.cpp)

set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)
add_subdirectory(${NAM_DEPS_PATH}/Catch2 Catch2 EXCLUDE_FROM_ALL)

set(TOOLS benchmodel)

add_custom_target(tools ALL
	DEPENDS ${TOOLS})

include_directories(tools ..)
include_directories(tools ${NAM_DEPS_PATH}/eigen)
include_directories(tools ${NAM_DEPS_PATH}/nlohmann)

add_executable(loadmodel loadmodel.cpp ${NAM_SOURCES})
add_executable(benchmodel benchmodel.cpp ${NAM_SOURCES})
add_executable(run_tests run_tests.cpp ${TEST_SOURCES} ${NAM_SOURCES})

target_link_libraries(run_tests PRIVATE Catch2::Catch2WithMain)

source_group(NAM ${CMAKE_CURRENT_SOURCE_DIR} FILES ${NAM_SOURCES})

target_compile_features(${TOOLS} PUBLIC cxx_std_20)
target_compile_features(run_tests PUBLIC cxx_std_20)

set_target_properties(${TOOLS}
	PROPERTIES
	CXX_VISIBILITY_PRESET hidden
	INTERPROCEDURAL_OPTIMIZATION TRUE
	PREFIX ""
)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	target_compile_definitions(${TOOLS} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
	target_compile_definitions(run_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

if (MSVC)
	target_compile_options(${TOOLS} PRIVATE
		"$<$<CONFIG:DEBUG>:/W4>"
		"$<$<CONFIG:RELEASE>:/O2>"
	)
	target_compile_options(run_tests PRIVATE
		"$<$<CONFIG:DEBUG>:/W4>"
		"$<$<CONFIG:RELEASE>:/O2>"
	)
else()
	target_compile_options(${TOOLS} PRIVATE
		-Wall -Wextra -Wpedantic -Wstrict-aliasing -Wunreachable-code -Weffc++ -Wno-unused-parameter
		"$<$<CONFIG:DEBUG>:-Og;-ggdb;-Werror>"
		"$<$<CONFIG:RELEASE>:-Ofast>"
	)
	target_compile_options(run_tests PRIVATE
		-Wall -Wextra -Wpedantic -Wstrict-aliasing -Wunreachable-code -Weffc++ -Wno-unused-parameter
		"$<$<CONFIG:DEBUG>:-Og;-ggdb;-Werror>"
		"$<$<CONFIG:RELEASE>:-Ofast>"
	)
endif()

set_source_files_properties(../NAM/dsp.cpp PROPERTIES COMPILE_FLAGS "-Wno-error")
