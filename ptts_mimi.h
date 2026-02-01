#ifndef PTTS_MIMI_H
#define PTTS_MIMI_H

#include <stdint.h>
#include "ptts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ptts_mimi ptts_mimi;

ptts_mimi *ptts_mimi_load(ptts_ctx *ctx);
void ptts_mimi_free(ptts_mimi *mm);

/*
 * Run a minimal Mimi decode stage: quantizer output projection + decoder transformer.
 * latent: length 32, output: length 512.
 */
int ptts_mimi_forward_one(ptts_mimi *mm, const float *latent, float *out_embed);

/* Decode a single FlowLM latent frame into raw audio (80 ms @ 24kHz). */
int ptts_mimi_decode_one(ptts_mimi *mm, const float *latent, float *out_audio, int *out_len);

/* Decode a sequence of FlowLM latents into raw audio. */
int ptts_mimi_decode(ptts_mimi *mm, const float *latents, int frames,
                     float *out_audio, int *out_len);

/* Encode raw audio into latents (512-dim continuous, unquantized).
 * audio: mono float array (24kHz assumed)
 * num_samples: length of audio
 * out_latents: pointer to receive malloc'd buffer of shape [frames, 512]
 * out_frames: receives number of frames
 */
int ptts_mimi_encode(ptts_mimi *mm, const float *audio, int num_samples,
                     float **out_latents, int *out_frames);

#ifdef __cplusplus
}
#endif

#endif /* PTTS_MIMI_H */
