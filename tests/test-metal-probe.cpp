#include "ggml-backend.h"

extern "C" GGML_BACKEND_API bool ggml_metal_test_probe_dispatch(void);

int main() {
    return ggml_metal_test_probe_dispatch() ? 0 : 1;
}
