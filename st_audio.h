#ifndef ST_AUDIO_H
#define ST_AUDIO_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int sample_rate;
    int channels;
    int num_samples; /* per channel */
    float *samples;  /* interleaved, length = num_samples * channels */
} st_audio;

st_audio *st_audio_create(int sample_rate, int channels, int num_samples);
void st_audio_free(st_audio *audio);

/* Save audio as 16-bit PCM WAV. Returns 0 on success, -1 on error. */
int st_audio_save_wav(const st_audio *audio, const char *path);

#ifdef __cplusplus
}
#endif

#endif /* ST_AUDIO_H */
