#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_API void ggml_critical_section_start(void);
GGML_API void ggml_critical_section_end(void);

GGML_API int ggml_tbq_norm_correction_enabled(void);

#ifdef __cplusplus
}
#endif
