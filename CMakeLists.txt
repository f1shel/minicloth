cmake_minimum_required(VERSION 3.16)
project(minicloth)

# Add your project files
file(GLOB SRC_FILES *.cpp *.h)
add_executable(${PROJECT_NAME} ${SRC_FILES})

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
option(LIBIGL_USE_STATIC_LIBRARY     "Use libIGL as static librarie" ON)
include(libigl)

# Enable the target igl::glfw
igl_include(glfw)
igl_include(stb)

# Link igl (and the glfw module) to your project
target_link_libraries(${PROJECT_NAME} PUBLIC
	igl::glfw
	igl::stb
)