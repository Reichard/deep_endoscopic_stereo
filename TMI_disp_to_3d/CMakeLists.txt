#########################
# Project
#########################

PROJECT (medikinfu-app)

#########################
# Include Directories
#########################

MESSAGE("${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES (
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/CUDA
)


#########################
# Add Executable
#########################

CUDA_ADD_EXECUTABLE (
        tmi
        tmi.cpp
        tmi.hpp
)

#########################
# Libraries
#########################

TARGET_LINK_LIBRARIES (
        tmi
        ${ITK_LIBRARIES}
        ${IVT_LIBRARIES}
        ${IGSTK_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${CUDA_LIBRARIES}
)
set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )
