# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /org/share/home/staff/reichard/workspace/depthlearning/tmi/viewer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /org/share/home/staff/reichard/workspace/depthlearning/tmi/viewer/build

# Include any dependencies generated for this target.
include CMakeFiles/viewer.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/viewer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/viewer.dir/flags.make

CMakeFiles/viewer.dir/main.cpp.o: CMakeFiles/viewer.dir/flags.make
CMakeFiles/viewer.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /org/share/home/staff/reichard/workspace/depthlearning/tmi/viewer/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/viewer.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/viewer.dir/main.cpp.o -c /org/share/home/staff/reichard/workspace/depthlearning/tmi/viewer/main.cpp

CMakeFiles/viewer.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/viewer.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /org/share/home/staff/reichard/workspace/depthlearning/tmi/viewer/main.cpp > CMakeFiles/viewer.dir/main.cpp.i

CMakeFiles/viewer.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/viewer.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /org/share/home/staff/reichard/workspace/depthlearning/tmi/viewer/main.cpp -o CMakeFiles/viewer.dir/main.cpp.s

CMakeFiles/viewer.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/viewer.dir/main.cpp.o.requires

CMakeFiles/viewer.dir/main.cpp.o.provides: CMakeFiles/viewer.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/viewer.dir/build.make CMakeFiles/viewer.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/viewer.dir/main.cpp.o.provides

CMakeFiles/viewer.dir/main.cpp.o.provides.build: CMakeFiles/viewer.dir/main.cpp.o

# Object files for target viewer
viewer_OBJECTS = \
"CMakeFiles/viewer.dir/main.cpp.o"

# External object files for target viewer
viewer_EXTERNAL_OBJECTS =

viewer: CMakeFiles/viewer.dir/main.cpp.o
viewer: CMakeFiles/viewer.dir/build.make
viewer: /usr/lib/x86_64-linux-gnu/libboost_system.so
viewer: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
viewer: /usr/lib/x86_64-linux-gnu/libboost_thread.so
viewer: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
viewer: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
viewer: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
viewer: /usr/lib/x86_64-linux-gnu/libpthread.so
viewer: /usr/local/lib/libpcl_common.so
viewer: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
viewer: /usr/local/lib/libpcl_kdtree.so
viewer: /usr/local/lib/libpcl_octree.so
viewer: /usr/local/lib/libpcl_search.so
viewer: /usr/local/lib/libpcl_surface.so
viewer: /usr/lib/libOpenNI.so
viewer: /usr/local/lib/libpcl_io.so
viewer: /usr/local/lib/libpcl_sample_consensus.so
viewer: /usr/local/lib/libpcl_filters.so
viewer: /usr/local/lib/libpcl_features.so
viewer: /usr/local/lib/libpcl_registration.so
viewer: /usr/local/lib/libpcl_ml.so
viewer: /usr/local/lib/libpcl_recognition.so
viewer: /usr/local/lib/libpcl_keypoints.so
viewer: /usr/local/lib/libpcl_cuda_sample_consensus.so
viewer: /usr/local/lib/libpcl_cuda_features.so
viewer: /usr/local/lib/libpcl_cuda_segmentation.so
viewer: /usr/local/lib/libpcl_visualization.so
viewer: /usr/local/lib/libpcl_segmentation.so
viewer: /usr/local/lib/libpcl_tracking.so
viewer: /usr/local/lib/libpcl_stereo.so
viewer: /usr/local/lib/libpcl_outofcore.so
viewer: /usr/local/lib/libpcl_gpu_containers.so
viewer: /usr/local/lib/libpcl_gpu_utils.so
viewer: /usr/local/lib/libpcl_gpu_surface.so
viewer: /usr/local/lib/libpcl_gpu_octree.so
viewer: /usr/local/lib/libpcl_gpu_segmentation.so
viewer: /usr/local/lib/libpcl_gpu_kinfu_large_scale.so
viewer: /usr/local/lib/libpcl_gpu_kinfu.so
viewer: /usr/local/lib/libpcl_gpu_features.so
viewer: /usr/lib/x86_64-linux-gnu/libboost_system.so
viewer: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
viewer: /usr/lib/x86_64-linux-gnu/libboost_thread.so
viewer: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
viewer: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
viewer: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
viewer: /usr/lib/x86_64-linux-gnu/libpthread.so
viewer: /usr/lib/libOpenNI.so
viewer: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
viewer: /usr/local/lib/vtk-5.10/libvtkGenericFiltering.so.5.10.1
viewer: /usr/local/lib/vtk-5.10/libvtkGeovis.so.5.10.1
viewer: /usr/local/lib/vtk-5.10/libvtkCharts.so.5.10.1
viewer: /usr/local/lib/libpcl_common.so
viewer: /usr/local/lib/libpcl_kdtree.so
viewer: /usr/local/lib/libpcl_octree.so
viewer: /usr/local/lib/libpcl_search.so
viewer: /usr/local/lib/libpcl_surface.so
viewer: /usr/local/lib/libpcl_io.so
viewer: /usr/local/lib/libpcl_sample_consensus.so
viewer: /usr/local/lib/libpcl_filters.so
viewer: /usr/local/lib/libpcl_features.so
viewer: /usr/local/lib/libpcl_registration.so
viewer: /usr/local/lib/libpcl_ml.so
viewer: /usr/local/lib/libpcl_recognition.so
viewer: /usr/local/lib/libpcl_keypoints.so
viewer: /usr/local/lib/libpcl_cuda_sample_consensus.so
viewer: /usr/local/lib/libpcl_cuda_features.so
viewer: /usr/local/lib/libpcl_cuda_segmentation.so
viewer: /usr/local/lib/libpcl_visualization.so
viewer: /usr/local/lib/libpcl_segmentation.so
viewer: /usr/local/lib/libpcl_tracking.so
viewer: /usr/local/lib/libpcl_stereo.so
viewer: /usr/local/lib/libpcl_outofcore.so
viewer: /usr/local/lib/libpcl_gpu_containers.so
viewer: /usr/local/lib/libpcl_gpu_utils.so
viewer: /usr/local/lib/libpcl_gpu_surface.so
viewer: /usr/local/lib/libpcl_gpu_octree.so
viewer: /usr/local/lib/libpcl_gpu_segmentation.so
viewer: /usr/local/lib/libpcl_gpu_kinfu_large_scale.so
viewer: /usr/local/lib/libpcl_gpu_kinfu.so
viewer: /usr/local/lib/libpcl_gpu_features.so
viewer: /usr/local/lib/vtk-5.10/libvtkViews.so.5.10.1
viewer: /usr/local/lib/vtk-5.10/libvtkInfovis.so.5.10.1
viewer: /usr/local/lib/vtk-5.10/libvtkWidgets.so.5.10.1
viewer: /usr/local/lib/vtk-5.10/libvtkVolumeRendering.so.5.10.1
viewer: /usr/local/lib/vtk-5.10/libvtkHybrid.so.5.10.1
viewer: /usr/local/lib/vtk-5.10/libvtkParallel.so.5.10.1
viewer: /usr/local/lib/vtk-5.10/libvtkRendering.so.5.10.1
viewer: /usr/local/lib/vtk-5.10/libvtkImaging.so.5.10.1
viewer: /usr/local/lib/vtk-5.10/libvtkGraphics.so.5.10.1
viewer: /usr/local/lib/vtk-5.10/libvtkIO.so.5.10.1
viewer: /usr/local/lib/vtk-5.10/libvtkFiltering.so.5.10.1
viewer: /usr/local/lib/vtk-5.10/libvtkCommon.so.5.10.1
viewer: /usr/local/lib/vtk-5.10/libvtksys.so.5.10.1
viewer: CMakeFiles/viewer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable viewer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/viewer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/viewer.dir/build: viewer
.PHONY : CMakeFiles/viewer.dir/build

CMakeFiles/viewer.dir/requires: CMakeFiles/viewer.dir/main.cpp.o.requires
.PHONY : CMakeFiles/viewer.dir/requires

CMakeFiles/viewer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/viewer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/viewer.dir/clean

CMakeFiles/viewer.dir/depend:
	cd /org/share/home/staff/reichard/workspace/depthlearning/tmi/viewer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /org/share/home/staff/reichard/workspace/depthlearning/tmi/viewer /org/share/home/staff/reichard/workspace/depthlearning/tmi/viewer /org/share/home/staff/reichard/workspace/depthlearning/tmi/viewer/build /org/share/home/staff/reichard/workspace/depthlearning/tmi/viewer/build /org/share/home/staff/reichard/workspace/depthlearning/tmi/viewer/build/CMakeFiles/viewer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/viewer.dir/depend

