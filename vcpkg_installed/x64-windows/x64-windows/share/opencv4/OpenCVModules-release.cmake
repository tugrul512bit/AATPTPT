#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "opencv_core" for configuration "Release"
set_property(TARGET opencv_core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_core PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/opencv_core4.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opencv_core4.dll"
  )

list(APPEND _cmake_import_check_targets opencv_core )
list(APPEND _cmake_import_check_files_for_opencv_core "${_IMPORT_PREFIX}/lib/opencv_core4.lib" "${_IMPORT_PREFIX}/bin/opencv_core4.dll" )

# Import target "opencv_flann" for configuration "Release"
set_property(TARGET opencv_flann APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_flann PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/opencv_flann4.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opencv_flann4.dll"
  )

list(APPEND _cmake_import_check_targets opencv_flann )
list(APPEND _cmake_import_check_files_for_opencv_flann "${_IMPORT_PREFIX}/lib/opencv_flann4.lib" "${_IMPORT_PREFIX}/bin/opencv_flann4.dll" )

# Import target "opencv_imgproc" for configuration "Release"
set_property(TARGET opencv_imgproc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_imgproc PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/opencv_imgproc4.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opencv_imgproc4.dll"
  )

list(APPEND _cmake_import_check_targets opencv_imgproc )
list(APPEND _cmake_import_check_files_for_opencv_imgproc "${_IMPORT_PREFIX}/lib/opencv_imgproc4.lib" "${_IMPORT_PREFIX}/bin/opencv_imgproc4.dll" )

# Import target "opencv_ml" for configuration "Release"
set_property(TARGET opencv_ml APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_ml PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/opencv_ml4.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opencv_ml4.dll"
  )

list(APPEND _cmake_import_check_targets opencv_ml )
list(APPEND _cmake_import_check_files_for_opencv_ml "${_IMPORT_PREFIX}/lib/opencv_ml4.lib" "${_IMPORT_PREFIX}/bin/opencv_ml4.dll" )

# Import target "opencv_photo" for configuration "Release"
set_property(TARGET opencv_photo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_photo PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/opencv_photo4.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opencv_photo4.dll"
  )

list(APPEND _cmake_import_check_targets opencv_photo )
list(APPEND _cmake_import_check_files_for_opencv_photo "${_IMPORT_PREFIX}/lib/opencv_photo4.lib" "${_IMPORT_PREFIX}/bin/opencv_photo4.dll" )

# Import target "opencv_dnn" for configuration "Release"
set_property(TARGET opencv_dnn APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_dnn PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/opencv_dnn4.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "protobuf::libprotobuf"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opencv_dnn4.dll"
  )

list(APPEND _cmake_import_check_targets opencv_dnn )
list(APPEND _cmake_import_check_files_for_opencv_dnn "${_IMPORT_PREFIX}/lib/opencv_dnn4.lib" "${_IMPORT_PREFIX}/bin/opencv_dnn4.dll" )

# Import target "opencv_features2d" for configuration "Release"
set_property(TARGET opencv_features2d APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_features2d PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/opencv_features2d4.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opencv_features2d4.dll"
  )

list(APPEND _cmake_import_check_targets opencv_features2d )
list(APPEND _cmake_import_check_files_for_opencv_features2d "${_IMPORT_PREFIX}/lib/opencv_features2d4.lib" "${_IMPORT_PREFIX}/bin/opencv_features2d4.dll" )

# Import target "opencv_imgcodecs" for configuration "Release"
set_property(TARGET opencv_imgcodecs APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_imgcodecs PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/opencv_imgcodecs4.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opencv_imgcodecs4.dll"
  )

list(APPEND _cmake_import_check_targets opencv_imgcodecs )
list(APPEND _cmake_import_check_files_for_opencv_imgcodecs "${_IMPORT_PREFIX}/lib/opencv_imgcodecs4.lib" "${_IMPORT_PREFIX}/bin/opencv_imgcodecs4.dll" )

# Import target "opencv_videoio" for configuration "Release"
set_property(TARGET opencv_videoio APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_videoio PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/opencv_videoio4.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opencv_videoio4.dll"
  )

list(APPEND _cmake_import_check_targets opencv_videoio )
list(APPEND _cmake_import_check_files_for_opencv_videoio "${_IMPORT_PREFIX}/lib/opencv_videoio4.lib" "${_IMPORT_PREFIX}/bin/opencv_videoio4.dll" )

# Import target "opencv_calib3d" for configuration "Release"
set_property(TARGET opencv_calib3d APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_calib3d PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/opencv_calib3d4.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opencv_calib3d4.dll"
  )

list(APPEND _cmake_import_check_targets opencv_calib3d )
list(APPEND _cmake_import_check_files_for_opencv_calib3d "${_IMPORT_PREFIX}/lib/opencv_calib3d4.lib" "${_IMPORT_PREFIX}/bin/opencv_calib3d4.dll" )

# Import target "opencv_highgui" for configuration "Release"
set_property(TARGET opencv_highgui APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_highgui PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/opencv_highgui4.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opencv_highgui4.dll"
  )

list(APPEND _cmake_import_check_targets opencv_highgui )
list(APPEND _cmake_import_check_files_for_opencv_highgui "${_IMPORT_PREFIX}/lib/opencv_highgui4.lib" "${_IMPORT_PREFIX}/bin/opencv_highgui4.dll" )

# Import target "opencv_objdetect" for configuration "Release"
set_property(TARGET opencv_objdetect APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_objdetect PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/opencv_objdetect4.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opencv_objdetect4.dll"
  )

list(APPEND _cmake_import_check_targets opencv_objdetect )
list(APPEND _cmake_import_check_files_for_opencv_objdetect "${_IMPORT_PREFIX}/lib/opencv_objdetect4.lib" "${_IMPORT_PREFIX}/bin/opencv_objdetect4.dll" )

# Import target "opencv_stitching" for configuration "Release"
set_property(TARGET opencv_stitching APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_stitching PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/opencv_stitching4.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opencv_stitching4.dll"
  )

list(APPEND _cmake_import_check_targets opencv_stitching )
list(APPEND _cmake_import_check_files_for_opencv_stitching "${_IMPORT_PREFIX}/lib/opencv_stitching4.lib" "${_IMPORT_PREFIX}/bin/opencv_stitching4.dll" )

# Import target "opencv_video" for configuration "Release"
set_property(TARGET opencv_video APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_video PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/opencv_video4.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opencv_video4.dll"
  )

list(APPEND _cmake_import_check_targets opencv_video )
list(APPEND _cmake_import_check_files_for_opencv_video "${_IMPORT_PREFIX}/lib/opencv_video4.lib" "${_IMPORT_PREFIX}/bin/opencv_video4.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
