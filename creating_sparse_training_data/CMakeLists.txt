
#########################
# Project
#########################

PROJECT (endoCheck)

 
#########################
# Include Directories
#########################

INCLUDE_DIRECTORIES (
	${CMAKE_SOURCE_DIR}/ImageProcessing/HRM
	${CUDA_SDK_ROOT_DIR}/common/inc
)


#########################
# Add Executable
#########################

QT4_WRAP_CPP (MOC_OUTFILES endoCheck.h)

CUDA_ADD_EXECUTABLE (
	endoCheck
	endoCheck.cpp
	${MOC_OUTFILES}
)

#########################
# Libraries
#########################

TARGET_LINK_LIBRARIES (
	endoCheck
	HRM
	${IVT_LIBRARIES}
	${IGSTK_LIBRARIES}
	${QT_LIBRARIES}
)
