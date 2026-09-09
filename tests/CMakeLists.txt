add_executable(test-mul-mat-convrot test-mul-mat-convrot.cpp)
target_link_libraries(test-mul-mat-convrot PRIVATE ggml ggml-cpu)

add_test(NAME test-mul-mat-convrot COMMAND test-mul-mat-convrot)
