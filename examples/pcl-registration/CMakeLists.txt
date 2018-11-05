#  minimum required cmake version: 2.1.0
cmake_minimum_required(VERSION 3.1.0)

project(RealSensePCLRegistrationMulticam)

# save the command line compile commands in the build output
set(cmake_export_compile_commands 1)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()


find_package( PCL REQUIRED )

if (PCL_FOUND)
    include_directories(${PCL_INCLUDE_DIRS})
    add_definitions(${PCL_DEFINITIONS})
    link_directories(${PCL_LIBRARY_DIRS})
endif()


set(DEPENDENCIES realsense2 ${PCL_LIBRARIES})

find_package(OpenGL)
if(NOT OPENGL_FOUND)
    message(FATAL_ERROR "\n\n OpenGL package is missing!\n\n")
endif()

list(APPEND DEPENDENCIES ${OPENGL_LIBRARIES})

if(WIN32)
    list(APPEND DEPENDENCIES glfw3)
else()
    # Find glfw header
    find_path(GLFW_INCLUDE_DIR NAMES GLFW/glfw3.h
            PATHS /usr/X11R6/include
            /usr/include/X11
            /opt/graphics/OpenGL/include
            /opt/graphics/OpenGL/contrib/libglfw
            /usr/local/include
            /usr/include/GL
            /usr/include
            )
    # Find glfw library
    find_library(GLFW_LIBRARIES NAMES glfw glfw3
            PATHS /usr/lib64
            /usr/lib
            /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
            /usr/local/lib64
            /usr/local/lib
            /usr/local/lib/${CMAKE_LIBRARY_ARCHITECTURE}
            /usr/X11R6/lib
            )
    list(APPEND DEPENDENCIES m ${GLFW_LIBRARIES} ${LIBUSB1_LIBRARIES})
    include_directories(${GLFW_INCLUDE_DIR})
endif()

add_executable(pcl-registration pcl-registration.cpp ../example.hpp)
target_link_libraries(pcl-registration ${DEPENDENCIES})
include_directories(pcl-registration ../)
set_target_properties (pcl-registration PROPERTIES
        FOLDER "Examples"
        )

install(
        TARGETS

        pcl-registration

        RUNTIME DESTINATION
        ${CMAKE_INSTALL_PREFIX}/bin
)