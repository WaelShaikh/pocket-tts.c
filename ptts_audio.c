#include "ptts_audio.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void write_u16_le(FILE *f, uint16_t v) {
    uint8_t b[2];
    b[0] = (uint8_t)(v & 0xff);
    b[1] = (uint8_t)((v >> 8) & 0xff);
    fwrite(b, 1, 2, f);
}

static void write_u32_le(FILE *f, uint32_t v) {
    uint8_t b[4];
    b[0] = (uint8_t)(v & 0xff);
    b[1] = (uint8_t)((v >> 8) & 0xff);
    b[2] = (uint8_t)((v >> 16) & 0xff);
    b[3] = (uint8_t)((v >> 24) & 0xff);
    fwrite(b, 1, 4, f);
}

static uint16_t read_u16_le(const uint8_t *b) {
    return (uint16_t)b[0] | ((uint16_t)b[1] << 8);
}

static uint32_t read_u32_le(const uint8_t *b) {
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

ptts_audio *ptts_audio_create(int sample_rate, int channels, int num_samples) {
    if (sample_rate <= 0 || channels <= 0 || num_samples < 0) return NULL;

    ptts_audio *audio = (ptts_audio *)calloc(1, sizeof(ptts_audio));
    if (!audio) return NULL;

    size_t total = (size_t)num_samples * (size_t)channels;
    audio->samples = (float *)calloc(total, sizeof(float));
    if (!audio->samples) {
        free(audio);
        return NULL;
    }

    audio->sample_rate = sample_rate;
    audio->channels = channels;
    audio->num_samples = num_samples;
    return audio;
}

void ptts_audio_free(ptts_audio *audio) {
    if (!audio) return;
    free(audio->samples);
    free(audio);
}

int ptts_audio_save_wav(const ptts_audio *audio, const char *path) {
    if (!audio || !audio->samples || !path) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    const uint16_t bits_per_sample = 16;
    const uint16_t bytes_per_sample = bits_per_sample / 8;
    const uint32_t num_channels = (uint32_t)audio->channels;
    const uint32_t sample_rate = (uint32_t)audio->sample_rate;
    const uint32_t total_samples = (uint32_t)audio->num_samples * num_channels;
    const uint32_t data_bytes = total_samples * bytes_per_sample;
    const uint32_t byte_rate = sample_rate * num_channels * bytes_per_sample;
    const uint16_t block_align = (uint16_t)(num_channels * bytes_per_sample);

    /* RIFF header */
    fwrite("RIFF", 1, 4, f);
    write_u32_le(f, 36 + data_bytes);
    fwrite("WAVE", 1, 4, f);

    /* fmt chunk */
    fwrite("fmt ", 1, 4, f);
    write_u32_le(f, 16);
    write_u16_le(f, 1); /* PCM */
    write_u16_le(f, (uint16_t)num_channels);
    write_u32_le(f, sample_rate);
    write_u32_le(f, byte_rate);
    write_u16_le(f, block_align);
    write_u16_le(f, bits_per_sample);

    /* data chunk */
    fwrite("data", 1, 4, f);
    write_u32_le(f, data_bytes);

    for (uint32_t i = 0; i < total_samples; i++) {
        float s = audio->samples[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        int16_t v = (int16_t)(s * 32767.0f);
        write_u16_le(f, (uint16_t)v);
    }

    fclose(f);
    return 0;
}

ptts_audio *ptts_audio_load_wav(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    uint8_t buf[12];
    if (fread(buf, 1, 12, f) != 12) { fclose(f); return NULL; }
    if (memcmp(buf, "RIFF", 4) != 0 || memcmp(buf + 8, "WAVE", 4) != 0) {
        fclose(f); return NULL;
    }

    /* Seek chunks */
    int channels = 0, sample_rate = 0, bits = 0;
    int data_found = 0;
    size_t data_len = 0;

    while (fread(buf, 1, 8, f) == 8) {
        uint32_t len = read_u32_le(buf + 4);
        if (memcmp(buf, "fmt ", 4) == 0) {
            uint8_t *fmt = malloc(len);
            if (fread(fmt, 1, len, f) != len) { free(fmt); fclose(f); return NULL; }
            uint16_t format = read_u16_le(fmt);
            channels = read_u16_le(fmt + 2);
            sample_rate = read_u32_le(fmt + 4);
            bits = read_u16_le(fmt + 14);
            free(fmt);
            if (format != 1 && format != 3) { /* PCM or IEEE Float */
                fclose(f); return NULL;
            }
        } else if (memcmp(buf, "data", 4) == 0) {
            data_found = 1;
            data_len = len;
            break; /* Assume data is last relevant chunk or we process it now */
        } else {
            fseek(f, len, SEEK_CUR);
        }
    }

    if (!data_found || channels == 0 || sample_rate == 0 || bits == 0) {
        fclose(f); return NULL;
    }

    int bytes_per_sample = bits / 8;
    int num_samples = data_len / (channels * bytes_per_sample);
    ptts_audio *audio = ptts_audio_create(sample_rate, channels, num_samples);
    if (!audio) { fclose(f); return NULL; }

    uint8_t *raw = malloc(data_len);
    if (fread(raw, 1, data_len, f) != data_len) {
        free(raw); ptts_audio_free(audio); fclose(f); return NULL;
    }
    fclose(f);

    float *out = audio->samples;
    if (bits == 16) {
        for (size_t i = 0; i < (size_t)num_samples * channels; i++) {
            int16_t v = (int16_t)read_u16_le(raw + i * 2);
            out[i] = (float)v / 32768.0f;
        }
    } else if (bits == 32) {
        /* Assume float if format==3, but some wavs use int32. Assuming int32 for PCM compatibility unless we parsed format strictly.
           Actually standard PCM wavs are usually 16 or 8. Let's assume standard int16.
           If bits is 32, it could be float. We won't support complex formats here for now. */
        // Simplified: support 16-bit only for now as it's standard
        free(raw); ptts_audio_free(audio); return NULL;
    } else {
        free(raw); ptts_audio_free(audio); return NULL;
    }

    free(raw);
    return audio;
}
