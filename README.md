# Supertonic 2 TTS (WIP)

A branch dedicated to implementing the Supertonic 2 TTS model by Supertone.

## Status

- **Pocket TTS implementation removed.**
- **Backend**: Generic C kernels and utilities (`st_*`) preserved and refactored.
- **Model**: Implementation pending (waiting for model details).

## Quick Start

```bash
# Build (CPU)
make cpu

# Run
./supertonic
```

## Structure

- `main.c`: Entry point.
- `st_kernels.c/h`: Compute kernels.
- `st_audio.c/h`: Audio utilities.
- `st_safetensors.c/h`: Safetensors reader.
- `st_spm.c/h`: SentencePiece tokenizer wrapper.

## License

MIT
