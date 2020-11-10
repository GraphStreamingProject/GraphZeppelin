# MODULE:
# FindOrDownloadPackage
#
# PROVIDES:
#   find_or_download_package(
#      PKG package
#      [VERSION version]
#      [SOURCE_DIR dir]
#      [TARGETS targets...]
#      [DOWNLOAD...]
#   )
#
# Inspired by https://github.com/Crascit/DownloadProject
# Tries to find_package(PKG VERSION CONFIG QUIET)
# If the package wasn't found, then downloads the package using the download options of ExternalProject-Add
# and adds the package using add_subdirectory. The targets specified are aliased to match the project.
function(find_or_download_package)
  #Save BUILD_SHARED_LIBS as a cached variable, in case a package screws with it *cough* xxHash *cough*
  set(FIND_OR_DOWNLOAD_SAVED_BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}" CACHE BOOL "" FORCE)

  #Parse arguments
  include(CMakeParseArguments)
  set(options "")
  set(oneValueArgs PKG VERSION SOURCE_DIR)
  set(multiValueArgs TARGETS DOWNLOAD)
  cmake_parse_arguments(FIND_OR_DOWNLOAD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  #Look for package normally
  find_package(${FIND_OR_DOWNLOAD_PKG} ${FIND_OR_DOWNLOAD_VERSION} CONFIG QUIET)
  #Detect if package was found by presence of all targets
  set(PKG_FOUND ON)
  foreach(PKG_TARGET IN LISTS FIND_OR_DOWNLOAD_TARGETS)
    if (NOT TARGET ${FIND_OR_DOWNLOAD_PKG}::${PKG_TARGET})
      set(PKG_FOUND OFF)
    endif()
  endforeach()

  if (PKG_FOUND)
    message("Found installation of ${FIND_OR_DOWNLOAD_PKG}, using it")
  else ()
    message("No installation of ${FIND_OR_DOWNLOAD_PKG} found, automatically downloading")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${FIND_OR_DOWNLOAD_PKG}-download/CMakeLists.txt
      "cmake_minimum_required(VERSION 2.8.11)\n"
      "project(${FIND_OR_DOWNLOAD_PKG}-download NONE)\n"
      "include(ExternalProject)\n"
      "ExternalProject_Add(${FIND_OR_DOWNLOAD_PKG}\n"
      "  ${FIND_OR_DOWNLOAD_DOWNLOAD}\n"
      "  SOURCE_DIR        \"${CMAKE_CURRENT_BINARY_DIR}/${FIND_OR_DOWNLOAD_PKG}-src\"\n"
      "  BINARY_DIR        \"${CMAKE_CURRENT_BINARY_DIR}/${FIND_OR_DOWNLOAD_PKG}-build\"\n"
      "  CONFIGURE_COMMAND \"\"\n"
      "  BUILD_COMMAND     \"\"\n"
      "  INSTALL_COMMAND   \"\"\n"
      "  TEST_COMMAND      \"\"\n"
      ")"
    )
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${FIND_OR_DOWNLOAD_PKG}-download
    )
    if(result)
      message(FATAL_ERROR "CMake step for ${FIND_OR_DOWNLOAD_PKG} failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${FIND_OR_DOWNLOAD_PKG}-download
    )
    if(result)
      message(FATAL_ERROR "Build step for ${FIND_OR_DOWNLOAD_PKG} failed: ${result}")
    endif()
  
    # Add package directly to current build
    add_subdirectory(
      ${CMAKE_CURRENT_BINARY_DIR}/${FIND_OR_DOWNLOAD_PKG}-src/${FIND_OR_DOWNLOAD_SOURCE_DIR}
      ${CMAKE_CURRENT_BINARY_DIR}/${FIND_OR_DOWNLOAD_PKG}-build/
      EXCLUDE_FROM_ALL
    )

    #Add aliases to targets
    foreach(PKG_TARGET IN LISTS FIND_OR_DOWNLOAD_TARGETS)
      add_library(${FIND_OR_DOWNLOAD_PKG}::${PKG_TARGET} ALIAS ${PKG_TARGET})
    endforeach()
  endif()
  
  #Restore BUILD_SHARED_LIBS
  set(BUILD_SHARED_LIBS "${FIND_OR_DOWNLOAD_SAVED_BUILD_SHARED_LIBS}" CACHE BOOL "" FORCE)
endfunction()