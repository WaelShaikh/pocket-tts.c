#ifndef ST_SPM_H
#define ST_SPM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct st_spm st_spm;

st_spm *st_spm_load(const char *path);
void st_spm_free(st_spm *spm);

/* Encode text into token IDs. Allocates *out_ids (caller frees). Returns 0 on success. */
int st_spm_encode(const st_spm *spm, const char *text, int **out_ids, int *out_len);

/* Access piece bytes for a given ID. Returns pointer owned by spm. */
const char *st_spm_piece(const st_spm *spm, int id, int *out_len);

int st_spm_vocab_size(const st_spm *spm);

#ifdef __cplusplus
}
#endif

#endif /* ST_SPM_H */
