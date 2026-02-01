#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "st_kernels.h"
#include "st_safetensors.h"
#include "st_audio.h"
#include "st_spm.h"

int main(int argc, char **argv) {
    printf("Supertonic 2 TTS Engine (WIP)\n");

    char *model_dir = "supertonic-model";
    if (argc > 1) {
        model_dir = argv[1];
    }

    printf("Checking model directory: %s\n", model_dir);

    const char *files[] = {
        "text_encoder.onnx",
        "duration_predictor.onnx",
        "vector_estimator.onnx",
        "vocoder.onnx",
        "tts.json",
        "unicode_indexer.json",
        NULL
    };

    int missing = 0;
    for (int i = 0; files[i]; i++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", model_dir, files[i]);
        if (access(path, F_OK) != 0) {
            printf("  [MISSING] %s\n", files[i]);
            missing++;
        } else {
            printf("  [OK]      %s\n", files[i]);
        }
    }

    if (missing) {
        printf("\nError: %d model files missing.\n", missing);
        printf("Please run: python3 download_model.py --out %s\n", model_dir);
        return 1;
    }

    printf("\nAll model files present.\n");
    printf("Note: This backend currently requires an ONNX Runtime implementation to execute the model.\n");
    printf("Use the generic 'st_*' utilities provided to build a C-based ONNX runner.\n");

    return 0;
}
