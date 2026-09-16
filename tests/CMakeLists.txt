add_executable(test-mul-mat-convrot test-mul-mat-convrot.cpp)
target_link_libraries(test-mul-mat-convrot PRIVATE ggml)
if (NOT GGML_BACKEND_DL)
    target_link_libraries(test-mul-mat-convrot PRIVATE ggml-cpu)
endif()

add_test(NAME test-mul-mat-convrot COMMAND test-mul-mat-convrot)

if (TARGET ggml-metal)
    add_executable(test-mul-mat-convrot-metal test-mul-mat-convrot-metal.cpp)
    target_link_libraries(test-mul-mat-convrot-metal PRIVATE ggml ggml-metal)
    add_test(NAME test-mul-mat-convrot-metal COMMAND test-mul-mat-convrot-metal)
endif()
