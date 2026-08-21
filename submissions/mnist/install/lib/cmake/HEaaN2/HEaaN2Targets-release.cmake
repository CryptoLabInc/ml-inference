#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "HEaaN2::HEaaN2" for configuration "Release"
set_property(TARGET HEaaN2::HEaaN2 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(HEaaN2::HEaaN2 PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libheaan2.so.0.2.0"
  IMPORTED_SONAME_RELEASE "libheaan2.so.0.2.0"
  )

list(APPEND _cmake_import_check_targets HEaaN2::HEaaN2 )
list(APPEND _cmake_import_check_files_for_HEaaN2::HEaaN2 "${_IMPORT_PREFIX}/lib/libheaan2.so.0.2.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
