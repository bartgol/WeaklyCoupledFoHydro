cmake_minimum_required(VERSION 2.7)
CMAKE_POLICY(SET CMP0060 NEW)

PROJECT (WeaklyCoupledFoHydro CXX)

IF (NOT DEFINED ALBANY_DIR)
  MESSAGE (FATAL_ERROR "Error! WeaklyCoupledFoHydro requires Albany.\n       Please, set the variable 'ALBANY_DIR' to point to a valid Albany installation.")
ENDIF()

SET(CMAKE_PREFIX_PATH ${ALBANY_DIR} ${CMAKE_PREFIX_PATH})

FIND_PACKAGE(Albany REQUIRED)

STRING (REPLACE ";" " " ALBANY_CXX_FLAGS "${Albany_CXX_FLAGS}")
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ALBANY_CXX_FLAGS}")

INCLUDE_DIRECTORIES (${Albany_INCLUDE_DIRS} ${Albany_TPL_INCLUDE_DIRS})
LINK_DIRECTORIES (${Albany_LIBRARY_DIRS} ${Albany_TPL_LIBRARY_DIRS})
ADD_EXECUTABLE (weakly_coupled_fo_hydro main.cpp)
TARGET_LINK_LIBRARIES (weakly_coupled_fo_hydro ${Albany_LIBRARIES} ${Albany_TPL_LIBRARIES})
