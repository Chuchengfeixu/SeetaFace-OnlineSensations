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
CMAKE_SOURCE_DIR = /home/dh/program/SeetaFaceEngine/FaceAlignment

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dh/program/SeetaFaceEngine/FaceAlignment/build

# Include any dependencies generated for this target.
include CMakeFiles/seeta_fa_lib.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/seeta_fa_lib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/seeta_fa_lib.dir/flags.make

CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.o: CMakeFiles/seeta_fa_lib.dir/flags.make
CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.o: ../src/cfan.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dh/program/SeetaFaceEngine/FaceAlignment/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.o -c /home/dh/program/SeetaFaceEngine/FaceAlignment/src/cfan.cpp

CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dh/program/SeetaFaceEngine/FaceAlignment/src/cfan.cpp > CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.i

CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dh/program/SeetaFaceEngine/FaceAlignment/src/cfan.cpp -o CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.s

CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.o.requires:

.PHONY : CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.o.requires

CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.o.provides: CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.o.requires
	$(MAKE) -f CMakeFiles/seeta_fa_lib.dir/build.make CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.o.provides.build
.PHONY : CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.o.provides

CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.o.provides.build: CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.o


CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.o: CMakeFiles/seeta_fa_lib.dir/flags.make
CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.o: ../src/face_alignment.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dh/program/SeetaFaceEngine/FaceAlignment/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.o -c /home/dh/program/SeetaFaceEngine/FaceAlignment/src/face_alignment.cpp

CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dh/program/SeetaFaceEngine/FaceAlignment/src/face_alignment.cpp > CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.i

CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dh/program/SeetaFaceEngine/FaceAlignment/src/face_alignment.cpp -o CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.s

CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.o.requires:

.PHONY : CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.o.requires

CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.o.provides: CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.o.requires
	$(MAKE) -f CMakeFiles/seeta_fa_lib.dir/build.make CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.o.provides.build
.PHONY : CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.o.provides

CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.o.provides.build: CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.o


CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.o: CMakeFiles/seeta_fa_lib.dir/flags.make
CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.o: ../src/sift.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dh/program/SeetaFaceEngine/FaceAlignment/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.o -c /home/dh/program/SeetaFaceEngine/FaceAlignment/src/sift.cpp

CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dh/program/SeetaFaceEngine/FaceAlignment/src/sift.cpp > CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.i

CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dh/program/SeetaFaceEngine/FaceAlignment/src/sift.cpp -o CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.s

CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.o.requires:

.PHONY : CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.o.requires

CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.o.provides: CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.o.requires
	$(MAKE) -f CMakeFiles/seeta_fa_lib.dir/build.make CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.o.provides.build
.PHONY : CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.o.provides

CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.o.provides.build: CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.o


# Object files for target seeta_fa_lib
seeta_fa_lib_OBJECTS = \
"CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.o" \
"CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.o" \
"CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.o"

# External object files for target seeta_fa_lib
seeta_fa_lib_EXTERNAL_OBJECTS =

libseeta_fa_lib.so: CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.o
libseeta_fa_lib.so: CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.o
libseeta_fa_lib.so: CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.o
libseeta_fa_lib.so: CMakeFiles/seeta_fa_lib.dir/build.make
libseeta_fa_lib.so: CMakeFiles/seeta_fa_lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dh/program/SeetaFaceEngine/FaceAlignment/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library libseeta_fa_lib.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/seeta_fa_lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/seeta_fa_lib.dir/build: libseeta_fa_lib.so

.PHONY : CMakeFiles/seeta_fa_lib.dir/build

CMakeFiles/seeta_fa_lib.dir/requires: CMakeFiles/seeta_fa_lib.dir/src/cfan.cpp.o.requires
CMakeFiles/seeta_fa_lib.dir/requires: CMakeFiles/seeta_fa_lib.dir/src/face_alignment.cpp.o.requires
CMakeFiles/seeta_fa_lib.dir/requires: CMakeFiles/seeta_fa_lib.dir/src/sift.cpp.o.requires

.PHONY : CMakeFiles/seeta_fa_lib.dir/requires

CMakeFiles/seeta_fa_lib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/seeta_fa_lib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/seeta_fa_lib.dir/clean

CMakeFiles/seeta_fa_lib.dir/depend:
	cd /home/dh/program/SeetaFaceEngine/FaceAlignment/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dh/program/SeetaFaceEngine/FaceAlignment /home/dh/program/SeetaFaceEngine/FaceAlignment /home/dh/program/SeetaFaceEngine/FaceAlignment/build /home/dh/program/SeetaFaceEngine/FaceAlignment/build /home/dh/program/SeetaFaceEngine/FaceAlignment/build/CMakeFiles/seeta_fa_lib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/seeta_fa_lib.dir/depend
