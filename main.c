#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "st_kernels.h"
#include "st_safetensors.h"
#include "st_audio.h"
#include "st_spm.h"

#ifdef ST_USE_ONNX
#include "onnxruntime_c_api.h"
#endif

int main(int argc, char **argv) {
    printf("Supertonic 2 TTS Engine (WIP)\n");

#ifdef ST_USE_ONNX
    printf("Backend: ONNX Runtime\n");
#else
    printf("Backend: Placeholder (No Inference)\n");
#endif

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

#ifdef ST_USE_ONNX
    // Initialize ONNX Runtime environment
    const OrtApi *g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "Error: Failed to get ONNX Runtime API\n");
        return 1;
    }

    OrtEnv *env;
    OrtStatus *status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "Supertonic", &env);
    if (status != NULL) {
        const char *msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error creating ORT Env: %s\n", msg);
        g_ort->ReleaseStatus(status);
        return 1;
    }
    printf("ONNX Runtime environment initialized successfully.\n");

    // Attempt to verify text_encoder.onnx
    OrtSessionOptions *session_options;
    g_ort->CreateSessionOptions(&session_options);
    g_ort->SetIntraOpNumThreads(session_options, 1);
    g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);

    OrtSession *session;
    char model_path[1024];
    snprintf(model_path, sizeof(model_path), "%s/text_encoder.onnx", model_dir);

    printf("Verifying load of %s...\n", model_path);
    status = g_ort->CreateSession(env, model_path, session_options, &session);

    if (status != NULL) {
        const char *msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error loading model: %s\n", msg);
        g_ort->ReleaseStatus(status);
        // Don't fail hard if model is broken/incompatible, just warn
        printf("Warning: Model verification failed. Architecture mismatch?\n");
    } else {
        printf("Model loaded successfully! Runtime is ready.\n");
        g_ort->ReleaseSession(session);
    }

    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    printf("\nNote: Full inference pipeline is not yet implemented in this C port.\n");
    printf("Use the C++ reference implementation for immediate TTS generation.\n");

#else
    printf("Note: This backend currently requires an ONNX Runtime implementation to execute the model.\n");
    printf("Use the generic 'st_*' utilities provided to build a C-based ONNX runner.\n");
    printf("To build with ONNX Runtime support:\n");
    printf("  1. ./download_ort.sh\n");
    printf("  2. make onnx\n");
#endif

    return 0;
}
