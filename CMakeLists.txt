cmake_minimum_required(VERSION 3.25)
project(cpp-template VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

string(APPEND CMAKE_CXX_FLAGS
      " -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wcast-qual -Wformat=2"
      " -Wundef -Wshadow -Wcast-align -Wunused -Wnull-dereference"
      " -Wdouble-promotion -Wimplicit-fallthrough -Wextra-semi -Woverloaded-virtual"
      " -Wnon-virtual-dtor -Wold-style-cast")

string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fsanitize=address,undefined -fno-omit-frame-pointer")
string(APPEND CMAKE_EXE_LINKER_FLAGS_DEBUG " -fsanitize=address,undefined -fno-omit-frame-pointer")

set(CMAKE_CTEST_ARGUMENTS "--output-on-failure")

if(BUILD_MAIN)
      add_executable(main-vmc main.cpp src/statistics.cpp)
      target_include_directories(main-vmc PRIVATE src include)
      target_link_libraries(main-vmc tbb atomic)
endif()

if(BUILD_TESTING)
      include(CTest)
      enable_testing()

      add_executable(test-ho tests/test-ho.cpp)
      target_include_directories(test-ho PRIVATE src include)
      target_link_libraries(test-ho tbb atomic)
      add_test(NAME test-ho COMMAND test-ho)

      add_executable(test-box tests/test-box.cpp)
      target_include_directories(test-box PRIVATE src include)
      target_link_libraries(test-box tbb atomic)
      add_test(NAME test-box COMMAND test-box)

      add_executable(test-stat tests/test-stat.cpp)
      target_include_directories(test-stat PRIVATE src include)
      target_link_libraries(test-stat tbb atomic)
      add_test(NAME test-stat COMMAND test-stat)
endif()
