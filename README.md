# Supertonic 2 TTS (WIP)

A branch dedicated to implementing the Supertonic 2 TTS model by Supertone.

## Status

- **Pocket TTS implementation removed.**
- **Backend**: Generic C kernels and utilities (`st_*`) preserved.
- **Runtime**: ONNX Runtime integration started.

## Requirements

- **ONNX Runtime**: Required for model execution.
- **Python**: Required for downloading models.

## Quick Start

### 1. Download Dependencies

```bash
# Download ONNX Runtime (Linux x64)
./download_ort.sh

# Download Model Weights
python3 download_model.py
```

### 2. Build

```bash
# Build with ONNX Runtime support
make onnx
```

### 3. Run

```bash
./supertonic
```

If successful, it will print "ONNX Runtime environment initialized successfully." and verify the model load.

## Structure

- `main.c`: Entry point (checks model presence, initializes ONNX Runtime).
- `st_kernels.c/h`: Compute kernels (legacy).
- `st_audio.c/h`: Audio utilities.
- `download_model.py`: Fetches weights from Hugging Face.
- `download_ort.sh`: Fetches ONNX Runtime library.

## License

MIT
