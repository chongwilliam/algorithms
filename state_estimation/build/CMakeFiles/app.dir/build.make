# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/william/utils/state_estimation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/william/utils/state_estimation/build

# Include any dependencies generated for this target.
include CMakeFiles/app.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/app.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/app.dir/flags.make

CMakeFiles/app.dir/test.cpp.o: CMakeFiles/app.dir/flags.make
CMakeFiles/app.dir/test.cpp.o: ../test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/william/utils/state_estimation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/app.dir/test.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/app.dir/test.cpp.o -c /home/william/utils/state_estimation/test.cpp

CMakeFiles/app.dir/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/app.dir/test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/william/utils/state_estimation/test.cpp > CMakeFiles/app.dir/test.cpp.i

CMakeFiles/app.dir/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/app.dir/test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/william/utils/state_estimation/test.cpp -o CMakeFiles/app.dir/test.cpp.s

CMakeFiles/app.dir/ukf.cpp.o: CMakeFiles/app.dir/flags.make
CMakeFiles/app.dir/ukf.cpp.o: ../ukf.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/william/utils/state_estimation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/app.dir/ukf.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/app.dir/ukf.cpp.o -c /home/william/utils/state_estimation/ukf.cpp

CMakeFiles/app.dir/ukf.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/app.dir/ukf.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/william/utils/state_estimation/ukf.cpp > CMakeFiles/app.dir/ukf.cpp.i

CMakeFiles/app.dir/ukf.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/app.dir/ukf.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/william/utils/state_estimation/ukf.cpp -o CMakeFiles/app.dir/ukf.cpp.s

CMakeFiles/app.dir/redis/RedisClient.cpp.o: CMakeFiles/app.dir/flags.make
CMakeFiles/app.dir/redis/RedisClient.cpp.o: ../redis/RedisClient.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/william/utils/state_estimation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/app.dir/redis/RedisClient.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/app.dir/redis/RedisClient.cpp.o -c /home/william/utils/state_estimation/redis/RedisClient.cpp

CMakeFiles/app.dir/redis/RedisClient.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/app.dir/redis/RedisClient.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/william/utils/state_estimation/redis/RedisClient.cpp > CMakeFiles/app.dir/redis/RedisClient.cpp.i

CMakeFiles/app.dir/redis/RedisClient.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/app.dir/redis/RedisClient.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/william/utils/state_estimation/redis/RedisClient.cpp -o CMakeFiles/app.dir/redis/RedisClient.cpp.s

CMakeFiles/app.dir/filter/ButterworthFilter.cpp.o: CMakeFiles/app.dir/flags.make
CMakeFiles/app.dir/filter/ButterworthFilter.cpp.o: ../filter/ButterworthFilter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/william/utils/state_estimation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/app.dir/filter/ButterworthFilter.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/app.dir/filter/ButterworthFilter.cpp.o -c /home/william/utils/state_estimation/filter/ButterworthFilter.cpp

CMakeFiles/app.dir/filter/ButterworthFilter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/app.dir/filter/ButterworthFilter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/william/utils/state_estimation/filter/ButterworthFilter.cpp > CMakeFiles/app.dir/filter/ButterworthFilter.cpp.i

CMakeFiles/app.dir/filter/ButterworthFilter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/app.dir/filter/ButterworthFilter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/william/utils/state_estimation/filter/ButterworthFilter.cpp -o CMakeFiles/app.dir/filter/ButterworthFilter.cpp.s

# Object files for target app
app_OBJECTS = \
"CMakeFiles/app.dir/test.cpp.o" \
"CMakeFiles/app.dir/ukf.cpp.o" \
"CMakeFiles/app.dir/redis/RedisClient.cpp.o" \
"CMakeFiles/app.dir/filter/ButterworthFilter.cpp.o"

# External object files for target app
app_EXTERNAL_OBJECTS =

app: CMakeFiles/app.dir/test.cpp.o
app: CMakeFiles/app.dir/ukf.cpp.o
app: CMakeFiles/app.dir/redis/RedisClient.cpp.o
app: CMakeFiles/app.dir/filter/ButterworthFilter.cpp.o
app: CMakeFiles/app.dir/build.make
app: /usr/lib/x86_64-linux-gnu/libhiredis.so
app: CMakeFiles/app.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/william/utils/state_estimation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable app"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/app.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/app.dir/build: app

.PHONY : CMakeFiles/app.dir/build

CMakeFiles/app.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/app.dir/cmake_clean.cmake
.PHONY : CMakeFiles/app.dir/clean

CMakeFiles/app.dir/depend:
	cd /home/william/utils/state_estimation/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/william/utils/state_estimation /home/william/utils/state_estimation /home/william/utils/state_estimation/build /home/william/utils/state_estimation/build /home/william/utils/state_estimation/build/CMakeFiles/app.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/app.dir/depend
