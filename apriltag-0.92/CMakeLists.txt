# CMakeLists.txt for apriltag, added by Jack Morrison 4/8/14. Not an
# original part of apriltag.
project(apriltag)
cmake_minimum_required(VERSION 2.8)

include(def_library)

add_definitions(-msse4.2 -pthread  -g -std=gnu99 -Wall -Wno-unused-parameter -Wno-format-zero-length -O3)

set(APRIL_SRCS
  apriltag.c
  image_f32.c
  image_u8.c
  image_u32.c
  unionfind.c
  zhash.c
  zarray.c
  matd.c
  homography.c
  graymodel.c
  tag36h11.c
  tag36h10.c
  segment2.c
  workerpool.c
  g2d.c
  )  

def_library(apriltag
  SOURCES ${APRIL_SRCS}
  LINK_LIBS m pthread
  )
