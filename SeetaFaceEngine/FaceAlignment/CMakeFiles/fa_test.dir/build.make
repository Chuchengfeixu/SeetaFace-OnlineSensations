# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_SOURCE_DIR = /home/dh/program/SeetaFaceEngine

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dh/program/SeetaFaceEngine

# Include any dependencies generated for this target.
include FaceAlignment/CMakeFiles/fa_test.dir/depend.make

# Include the progress variables for this target.
include FaceAlignment/CMakeFiles/fa_test.dir/progress.make

# Include the compile flags for this target's objects.
include FaceAlignment/CMakeFiles/fa_test.dir/flags.make

FaceAlignment/CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.o: FaceAlignment/CMakeFiles/fa_test.dir/flags.make
FaceAlignment/CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.o: FaceAlignment/src/test/face_alignment_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dh/program/SeetaFaceEngine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object FaceAlignment/CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.o"
	cd /home/dh/program/SeetaFaceEngine/FaceAlignment && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.o -c /home/dh/program/SeetaFaceEngine/FaceAlignment/src/test/face_alignment_test.cpp

FaceAlignment/CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.i"
	cd /home/dh/program/SeetaFaceEngine/FaceAlignment && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dh/program/SeetaFaceEngine/FaceAlignment/src/test/face_alignment_test.cpp > CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.i

FaceAlignment/CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.s"
	cd /home/dh/program/SeetaFaceEngine/FaceAlignment && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dh/program/SeetaFaceEngine/FaceAlignment/src/test/face_alignment_test.cpp -o CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.s

FaceAlignment/CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.o.requires:

.PHONY : FaceAlignment/CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.o.requires

FaceAlignment/CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.o.provides: FaceAlignment/CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.o.requires
	$(MAKE) -f FaceAlignment/CMakeFiles/fa_test.dir/build.make FaceAlignment/CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.o.provides.build
.PHONY : FaceAlignment/CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.o.provides

FaceAlignment/CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.o.provides.build: FaceAlignment/CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.o


# Object files for target fa_test
fa_test_OBJECTS = \
"CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.o"

# External object files for target fa_test
fa_test_EXTERNAL_OBJECTS =

FaceAlignment/fa_test: FaceAlignment/CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.o
FaceAlignment/fa_test: FaceAlignment/CMakeFiles/fa_test.dir/build.make
FaceAlignment/fa_test: FaceAlignment/libseeta_fa_lib.so
FaceAlignment/fa_test: /usr/local/lib/libopencv_viz.so.3.1.0
FaceAlignment/fa_test: /usr/local/lib/libopencv_videostab.so.3.1.0
FaceAlignment/fa_test: /usr/local/lib/libopencv_superres.so.3.1.0
FaceAlignment/fa_test: /usr/local/lib/libopencv_stitching.so.3.1.0
FaceAlignment/fa_test: /usr/local/lib/libopencv_shape.so.3.1.0
FaceAlignment/fa_test: /usr/local/lib/libopencv_photo.so.3.1.0
FaceAlignment/fa_test: /usr/local/lib/libopencv_objdetect.so.3.1.0
FaceAlignment/fa_test: /usr/local/lib/libopencv_calib3d.so.3.1.0
FaceAlignment/fa_test: FaceDetection/libseeta_facedet_lib.so
FaceAlignment/fa_test: /usr/local/lib/libopencv_features2d.so.3.1.0
FaceAlignment/fa_test: /usr/local/lib/libopencv_ml.so.3.1.0
FaceAlignment/fa_test: /usr/local/lib/libopencv_highgui.so.3.1.0
FaceAlignment/fa_test: /usr/local/lib/libopencv_videoio.so.3.1.0
FaceAlignment/fa_test: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
FaceAlignment/fa_test: /usr/local/lib/libopencv_flann.so.3.1.0
FaceAlignment/fa_test: /usr/local/lib/libopencv_video.so.3.1.0
FaceAlignment/fa_test: /usr/local/lib/libopencv_imgproc.so.3.1.0
FaceAlignment/fa_test: /usr/local/lib/libopencv_core.so.3.1.0
FaceAlignment/fa_test: FaceAlignment/CMakeFiles/fa_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dh/program/SeetaFaceEngine/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable fa_test"
	cd /home/dh/program/SeetaFaceEngine/FaceAlignment && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fa_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
FaceAlignment/CMakeFiles/fa_test.dir/build: FaceAlignment/fa_test

.PHONY : FaceAlignment/CMakeFiles/fa_test.dir/build

FaceAlignment/CMakeFiles/fa_test.dir/requires: FaceAlignment/CMakeFiles/fa_test.dir/src/test/face_alignment_test.cpp.o.requires

.PHONY : FaceAlignment/CMakeFiles/fa_test.dir/requires

FaceAlignment/CMakeFiles/fa_test.dir/clean:
	cd /home/dh/program/SeetaFaceEngine/FaceAlignment && $(CMAKE_COMMAND) -P CMakeFiles/fa_test.dir/cmake_clean.cmake
.PHONY : FaceAlignment/CMakeFiles/fa_test.dir/clean

FaceAlignment/CMakeFiles/fa_test.dir/depend:
	cd /home/dh/program/SeetaFaceEngine && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dh/program/SeetaFaceEngine /home/dh/program/SeetaFaceEngine/FaceAlignment /home/dh/program/SeetaFaceEngine /home/dh/program/SeetaFaceEngine/FaceAlignment /home/dh/program/SeetaFaceEngine/FaceAlignment/CMakeFiles/fa_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : FaceAlignment/CMakeFiles/fa_test.dir/depend

