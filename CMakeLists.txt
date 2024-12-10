cmake_minimum_required(VERSION 3.25)
project(cpp-template VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Set the "default" value of the variables
if(NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE Debug)
endif()

# Add compilation flags
string(APPEND CMAKE_CXX_FLAGS
      " -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wcast-qual -Wformat=2"
      " -Wundef -Wshadow -Wcast-align -Wunused -Wnull-dereference"
      " -Wdouble-promotion -Wimplicit-fallthrough -Wextra-semi -Woverloaded-virtual"
      " -Wnon-virtual-dtor -Wold-style-cast")
string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fsanitize=address,undefined -fno-omit-frame-pointer")
string(APPEND CMAKE_EXE_LINKER_FLAGS_DEBUG " -fsanitize=address,undefined -fno-omit-frame-pointer")
set(CMAKE_CTEST_ARGUMENTS "--output-on-failure")

# If BUILDT_ALL is ON, set all BUILDT variables to ON
set(BUILDT_VARIABLES "")
list(APPEND BUILDT_VARIABLES BUILDT_HO_1P1D BUILDT_HO_1P2D BUILDT_HO_2P1D BUILDT_BOX_1P1D BUILDT_RAD_1P1D BUILDT_STAT)
foreach(X IN LISTS BUILDT_VARIABLES)
      if(BUILDT_ALL)
            set("${X}" ON)
      endif()
endforeach()

# Build main
if(BUILD_MAIN)
      add_executable(main-vmc main.cpp src/statistics.cpp)
      target_include_directories(main-vmc PRIVATE src include)
      target_link_libraries(main-vmc tbb atomic)
endif()

# Build tests
if(BUILDT_HO_1P1D OR BUILDT_HO_1P2D OR BUILDT_HO_2P1D OR BUILDT_BOX_1P1D OR BUILDT_RAD_1P1D OR BUILDT_STAT)
      include(CTest)
      enable_testing()
endif()
if(BUILDT_HO_1P1D)
      add_executable(test-ho-1p1d tests/test-ho-1p1d.cpp)
      target_include_directories(test-ho-1p1d PRIVATE src include)
      target_link_libraries(test-ho-1p1d tbb atomic)
      add_test(NAME test-ho-1p1d COMMAND test-ho-1p1d)
endif()
if(BUILDT_HO_1P2D)
      add_executable(test-ho-1p2d tests/test-ho-1p2d.cpp)
      target_include_directories(test-ho-1p2d PRIVATE src include)
      target_link_libraries(test-ho-1p2d tbb atomic)
      add_test(NAME test-ho-1p2d COMMAND test-ho-1p2d)
endif()
if(BUILDT_HO_2P1D)
      add_executable(test-ho-2p1d tests/test-ho-2p1d.cpp)
      target_include_directories(test-ho-2p1d PRIVATE src include)
      target_link_libraries(test-ho-2p1d tbb atomic)
      add_test(NAME test-ho-2p1d COMMAND test-ho-2p1d)
endif()
if(BUILDT_BOX_1P1D)
      add_executable(test-box-1p1d tests/test-box-1p1d.cpp)
      target_include_directories(test-box-1p1d PRIVATE src include)
      target_link_libraries(test-box-1p1d tbb atomic)
      add_test(NAME test-box-1p1d COMMAND test-box-1p1d)
endif()
if(BUILDT_RAD_1P1D)
      add_executable(test-rad-1p1d tests/test-rad-1p1d.cpp)
      target_include_directories(test-rad-1p1d PRIVATE src include)
      target_link_libraries(test-rad-1p1d tbb atomic)
      add_test(NAME test-rad-1p1d COMMAND test-rad-1p1d)
endif()
if(BUILDT_STAT)
      add_executable(test-stat tests/test-stat.cpp)
      target_include_directories(test-stat PRIVATE src include)
      target_link_libraries(test-stat tbb atomic)
      add_test(NAME test-stat COMMAND test-stat)
endif()
