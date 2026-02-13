set(TOOLS
	loadmodel
	benchmodel
	c_api_example
	run_tests
)

add_custom_target(tools ALL DEPENDS ${TOOLS})

add_executable(loadmodel loadmodel.cpp)
add_executable(benchmodel benchmodel.cpp)
add_executable(c_api_example c_api_example.cpp)
add_executable(run_tests run_tests.cpp test/allocation_tracking.cpp)

foreach(target ${TOOLS})
	target_link_libraries(${target} PRIVATE nam_static)
	target_include_directories(${target} PRIVATE
		..
		${NAM_DEPS_PATH}/eigen
		${NAM_DEPS_PATH}/nlohmann
	)
endforeach()
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

foreach(target ${TOOLS})
	target_compile_features(${target} PUBLIC cxx_std_20)
	set_target_properties(${target}
		PROPERTIES
		CXX_VISIBILITY_PRESET hidden
		INTERPROCEDURAL_OPTIMIZATION TRUE
		PREFIX ""
	)
	if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
		target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
	endif()
	if (MSVC)
		target_compile_options(${target} PRIVATE
			"$<$<CONFIG:DEBUG>:/W4>"
			"$<$<CONFIG:RELEASE>:/O2>"
		)
	else()
		target_compile_options(${target} PRIVATE
			-Wall -Wextra -Wpedantic -Wstrict-aliasing -Wunreachable-code -Weffc++ -Wno-unused-parameter
			"$<$<CONFIG:DEBUG>:-Og;-ggdb;-Werror>"
			"$<$<CONFIG:RELEASE>:-Ofast>"
		)
	endif()
endforeach()

install(TARGETS ${TOOLS}
	RUNTIME DESTINATION bin
	COMPONENT tools
)
