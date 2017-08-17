cmake_minimum_required(VERSION 2.8.8)
project(SpTRSV)

message(STATUS "System is " ${CMAKE_SYSTEM_NAME})
message(STATUS "Compiler is " ${CMAKE_CXX_COMPILER_ID})
message(STATUS "Source dir is " ${CMAKE_SOURCE_DIR})
message(STATUS "Source dir is " ${CMAKE_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 14)

message(STATUS "CXX Flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "Linker Flags: " ${CMAKE_EXE_LINKER_FLAGS})
set(dir ${CMAKE_CURRENT_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}")

set(SOURCE_FILES
    docopt/docopt.cpp
    main.cpp
    profiler.cpp
)

set(HEADER_FILES
    docopt/docopt.h
    docopt/docopt_private.h
    docopt/docopt_util.h
    docopt/docopt_value.h
    profiler.h
)

add_executable(sptrsv ${SOURCE_FILES} ${HEADER_FILES})

##
## MMMatrixIO library
##
include_directories(${CMAKE_SOURCE_DIR}/../MMMatrixIO/src)
find_library(MMMatrixIO
  mmmatrixio
  PATHS ${CMAKE_SOURCE_DIR}/../MMMatrixIO/build
)
target_link_libraries(sptrsv ${MMMatrixIO})


##
## MKL
##
set(MKL_ROOT_DIR $ENV{MKLROOT})
if (MKL_ROOT_DIR)
  message(STATUS "Found MKL at " $ENV{MKLROOT})
  add_definitions(-DMKL_EXISTS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64" )
  include_directories(${MKL_ROOT_DIR}/include)
  find_library(MKL_CORE
    mkl_core
    PATHS ${MKL_ROOT_DIR}/lib ${MKL_ROOT_DIR}/lib/intel64
  )
  find_library(MKL_INTEL_THREAD
    mkl_intel_thread
    PATHS ${MKL_ROOT_DIR}/lib ${MKL_ROOT_DIR}/lib/intel64
  )
  find_library(MKL_INTEL_LP64
    mkl_intel_lp64
    PATHS ${MKL_ROOT_DIR}/lib ${MKL_ROOT_DIR}/lib/intel64
  )
  find_library(MKL_INTEL_LP64
    iomp5
    PATHS ${MKL_ROOT_DIR}/../lib ${MKL_ROOT_DIR}/../lib/intel64
  )
  target_link_libraries(sptrsv
    ${MKL_CORE}
    ${MKL_INTEL_THREAD}
    ${MKL_INTEL_LP64}
    ${IOMP5}
    pthread
    m
    dl
  )
endif()


##
## OpenMP
##
find_package(OpenMP)
if (OPENMP_FOUND)
  add_definitions(-DOPENMP_EXISTS)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message(STATUS "System is " ${CMAKE_SYSTEM_NAME})
endif()

