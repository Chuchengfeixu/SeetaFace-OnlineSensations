aux_source_directory (. SRC_LIST)

message(${SRC_LIST})

# add external libraries
find_package(OpenCV REQUIRED)

enable_testing ()
foreach (f ${SRC_LIST})
  string(REGEX REPLACE "[.]cpp" ".bin" BIN ${f})
  add_executable(${BIN} ${f})
  target_link_libraries(${BIN} viplnet ${OpenCV_LIBS})
  target_link_libraries(${BIN} viplnet "/home/dh/program/SeetaFaceEngine/FaceIdentification/lib/libseeta_facedet_lib.so") 
  target_link_libraries(${BIN} viplnet "/home/dh/program/SeetaFaceEngine/FaceIdentification/lib/libseeta_fa_lib.so") 
endforeach ()

