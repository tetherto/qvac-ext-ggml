cmake_minimum_required(VERSION 3.16)

file(READ "${GGML_SOURCE}/src/ggml-xdna/xdna-runtime.cpp" RUNTIME)
string(FIND "${RUNTIME}" "static fs::path xdna_module_dir(void) {" START)
string(FIND "${RUNTIME}" "\nstd::vector<fs::path> xdna_kernel_search_dirs" END)
math(EXPR LENGTH "${END} - ${START}")
string(SUBSTRING "${RUNTIME}" ${START} ${LENGTH} MODULE_DIR)

file(READ "${GGML_SOURCE}/src/ggml-xdna/CMakeLists.txt" BACKEND_CMAKE)
string(FIND "${BACKEND_CMAKE}" "    if (GGML_BACKEND_DL AND GGML_BACKEND_DIR)" START)
string(FIND "${BACKEND_CMAKE}" " COMPONENT ggml-xdna)" END)
string(LENGTH " COMPONENT ggml-xdna)" END_LENGTH)
math(EXPR LENGTH "${END} + ${END_LENGTH} - ${START}")
string(SUBSTRING "${BACKEND_CMAKE}" ${START} ${LENGTH} INSTALL_ARTIFACTS)

set(PROJECT_TEMPLATE [=[
cmake_minimum_required(VERSION 3.16)
project(xdna_artifact_install CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_BINDIR bin)
set(BUILD_SHARED_LIBS ON)
set(GGML_BACKEND_DL OFF)
add_library(backend SHARED library.cpp)
target_link_libraries(backend PRIVATE ${CMAKE_DL_LIBS})
install(TARGETS backend LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
set(XDNA_INSTALL_ARTIFACTS "${CMAKE_CURRENT_SOURCE_DIR}/sample.xclbin")
@SELECTION@
add_executable(probe probe.cpp)
target_link_libraries(probe PRIVATE ${CMAKE_DL_LIBS})
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/backend-name.txt" CONTENT "$<TARGET_FILE_NAME:backend>")
]=])

file(MAKE_DIRECTORY "${TEST_ROOT}/outside")
foreach(CASE IN ITEMS before after)
    set(CASE_ROOT "${TEST_ROOT}/${CASE}")
    file(MAKE_DIRECTORY "${CASE_ROOT}")
    file(WRITE "${CASE_ROOT}/library.cpp"
        "#include <filesystem>\n#include <system_error>\n#include <dlfcn.h>\nnamespace fs = std::filesystem;\n${MODULE_DIR}\nextern \"C\" int artifact_found() { return fs::is_regular_file(xdna_module_dir() / \"sample.xclbin\") ? 0 : 2; }\n")
    file(WRITE "${CASE_ROOT}/probe.cpp" [=[
#include <dlfcn.h>
int main(int argc, char ** argv) {
    if (argc != 2) return 10;
    void * library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) return 11;
    auto check = reinterpret_cast<int (*)()>(dlsym(library, "artifact_found"));
    if (!check) return 12;
    const int result = check();
    dlclose(library);
    return result;
}
]=])
    file(WRITE "${CASE_ROOT}/sample.xclbin" "artifact installation sentinel\n")
    set(SELECTION "${INSTALL_ARTIFACTS}")
    if (CASE STREQUAL "before")
        string(REPLACE "GGML_BACKEND_DL OR BUILD_SHARED_LIBS" "GGML_BACKEND_DL" SELECTION "${SELECTION}")
        set(EXPECTED 2)
    else()
        set(EXPECTED 0)
    endif()
    string(CONFIGURE "${PROJECT_TEMPLATE}" PROJECT @ONLY)
    file(WRITE "${CASE_ROOT}/CMakeLists.txt" "${PROJECT}")
    execute_process(COMMAND "${CMAKE_COMMAND}" -S "${CASE_ROOT}" -B "${CASE_ROOT}/build"
        "-DCMAKE_INSTALL_PREFIX=${CASE_ROOT}/prefix"
        RESULT_VARIABLE RESULT OUTPUT_FILE "${CASE_ROOT}/configure.log" ERROR_FILE "${CASE_ROOT}/configure.err")
    if (NOT RESULT EQUAL 0)
        message(FATAL_ERROR "${CASE}: configure failed; see ${CASE_ROOT}/configure.err")
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" --build "${CASE_ROOT}/build"
        RESULT_VARIABLE RESULT OUTPUT_FILE "${CASE_ROOT}/build.log" ERROR_FILE "${CASE_ROOT}/build.err")
    if (NOT RESULT EQUAL 0)
        message(FATAL_ERROR "${CASE}: build failed; see ${CASE_ROOT}/build.err")
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" --install "${CASE_ROOT}/build"
        RESULT_VARIABLE RESULT OUTPUT_FILE "${CASE_ROOT}/install.log" ERROR_FILE "${CASE_ROOT}/install.err")
    if (NOT RESULT EQUAL 0)
        message(FATAL_ERROR "${CASE}: install failed; see ${CASE_ROOT}/install.err")
    endif()
    file(READ "${CASE_ROOT}/build/backend-name.txt" LIBRARY_NAME)
    execute_process(COMMAND "${CASE_ROOT}/build/probe" "${CASE_ROOT}/prefix/lib/${LIBRARY_NAME}"
        WORKING_DIRECTORY "${TEST_ROOT}/outside" RESULT_VARIABLE RESULT)
    if (NOT RESULT EQUAL EXPECTED)
        message(FATAL_ERROR "${CASE}: outside-prefix artifact discovery returned ${RESULT}; expected ${EXPECTED}")
    endif()
endforeach()
