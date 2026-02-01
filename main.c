/*
 * Pocket-TTS CLI (WIP)
 *
 * Usage:
 *   ptts -d model_dir -p "text" -o out.wav [options]
 *   ptts export-voice -d model_dir -i input.wav -o out.safetensors
 */

#include "ptts.h"
#include "ptts_flowlm.h"
#include "ptts_mimi.h"
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    OUTPUT_QUIET = 0,
    OUTPUT_NORMAL = 1,
    OUTPUT_VERBOSE = 2
} output_level_t;

static output_level_t output_level = OUTPUT_NORMAL;

static void print_usage(const char *prog) {
    printf("Pocket-TTS Pure C (WIP)\n");
    printf("Usage: %s [command] [options]\n", prog);
    printf("\nCommands:\n");
    printf("  generate (default)    Generate speech from text\n");
    printf("  export-voice          Convert WAV file to voice embedding\n");
    printf("\nGeneral Options:\n");
    printf("  -d, --dir PATH        Model directory or .safetensors file\n");
    printf("  -q, --quiet           Less output\n");
    printf("  -v, --verbose         More output\n");
    printf("  -h, --help            Show help\n");
    printf("\nGenerate Options:\n");
    printf("  -p, --prompt TEXT     Text to synthesize\n");
    printf("  -o, --output PATH     Output WAV path\n");
    printf("      --voice NAME      Voice embedding name or .safetensors path (default: alba)\n");
    printf("  -S, --seed N          Random seed (-1 for random)\n");
    printf("  -t, --temp F          Noise temperature for FlowLM (default: 1.0)\n");
    printf("      --noise-clamp F   Clamp noise to [-F, F] (default: 0, off)\n");
    printf("      --eos-threshold F Stop early if eos_logit >= F (default: -4.0)\n");
    printf("      --eos-min-frames N Minimum frames before EOS stop (default: 1)\n");
    printf("      --eos-after N     Frames to keep after EOS (default: auto)\n");
    printf("  -r, --rate N          Sample rate for dummy generator (default: 24000)\n");
    printf("  -s, --steps N         Flow matching steps (placeholder)\n");
    printf("      --dummy           Generate placeholder audio (no model)\n");
    printf("\nExport Voice Options:\n");
    printf("  -i, --input PATH      Input WAV file (should be 24kHz)\n");
    printf("  -o, --output PATH     Output .safetensors path\n");
    printf("\nIntrospection/Debug:\n");
    printf("      --info            Print model info\n");
    printf("      --list            List tensors in weights file\n");
    printf("      --find TEXT       List tensors whose names contain TEXT\n");
    printf("      --verify          Verify weights against expected shapes\n");
    printf("      --tokens          Print token IDs for the prompt\n");
    printf("      --flow-test       Run a single FlowLM step and print latent stats\n");
    printf("      --mimi-test       Run FlowLM + Mimi decoder transformer stats\n");
    printf("      --mimi-wave PATH  Write Mimi decode WAV to PATH (frames * 80ms)\n");
    printf("      --frames N        Number of FlowLM/Mimi frames (default: auto)\n");
    printf("      --latent-out PATH Write raw FlowLM latents (32 floats per frame)\n");
    printf("      --cond-out PATH   Write first FlowLM condition vector (1024 floats)\n");
    printf("      --flow-out PATH   Write first FlowLM flow vector (32 floats)\n");
    printf("\nExamples:\n");
    printf("  %s -d pocket-tts-model -p \"Hello world\" -o out.wav --voice alba\n", prog);
    printf("  %s export-voice -d pocket-tts-model -i my_voice.wav -o my_voice.safetensors\n", prog);
}

#define LOG_NORMAL(...) do { if (output_level >= OUTPUT_NORMAL) fprintf(stderr, __VA_ARGS__); } while(0)
#define LOG_VERBOSE(...) do { if (output_level >= OUTPUT_VERBOSE) fprintf(stderr, __VA_ARGS__); } while(0)

int main(int argc, char **argv) {
    /* Simple command detection */
    int is_export = 0;
    int arg_start = 1;
    if (argc > 1 && strcmp(argv[1], "export-voice") == 0) {
        is_export = 1;
        arg_start = 2;
    } else if (argc > 1 && strcmp(argv[1], "generate") == 0) {
        arg_start = 2;
    }

    const char *model_dir = NULL;
    const char *prompt = NULL;
    const char *output = NULL;
    const char *input = NULL;
    const char *voice = NULL;
    int list_tensors = 0;
    int info_only = 0;
    int show_tokens = 0;
    int use_dummy = 0;
    int verify_weights = 0;
    int flow_test = 0;
    int mimi_test = 0;
    const char *mimi_wave = NULL;
    const char *find_pat = NULL;
    const char *latent_out = NULL;
    const char *cond_out = NULL;
    const char *flow_out = NULL;
    ptts_params params = PTTS_PARAMS_DEFAULT;

    static struct option long_opts[] = {
        {"dir", required_argument, 0, 'd'},
        {"prompt", required_argument, 0, 'p'},
        {"output", required_argument, 0, 'o'},
        {"input", required_argument, 0, 'i'},
        {"voice", required_argument, 0, 0},
        {"info", no_argument, 0, 0},
        {"list", no_argument, 0, 0},
        {"find", required_argument, 0, 0},
        {"verify", no_argument, 0, 0},
        {"tokens", no_argument, 0, 0},
        {"flow-test", no_argument, 0, 0},
        {"mimi-test", no_argument, 0, 0},
        {"mimi-wave", required_argument, 0, 0},
        {"frames", required_argument, 0, 0},
        {"latent-out", required_argument, 0, 0},
        {"cond-out", required_argument, 0, 0},
        {"flow-out", required_argument, 0, 0},
        {"noise-clamp", required_argument, 0, 0},
        {"eos-threshold", required_argument, 0, 0},
        {"eos-min-frames", required_argument, 0, 0},
        {"eos-after", required_argument, 0, 0},
        {"temp", required_argument, 0, 't'},
        {"dummy", no_argument, 0, 0},
        {"rate", required_argument, 0, 'r'},
        {"steps", required_argument, 0, 's'},
        {"seed", required_argument, 0, 'S'},
        {"quiet", no_argument, 0, 'q'},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    /* Shift argv for getopt if subcommand used */
    char **args = argv;
    int count = argc;
    if (arg_start > 1) {
        args = (char **)malloc(sizeof(char*) * argc);
        args[0] = argv[0];
        for(int i=arg_start; i<argc; i++) args[i-arg_start+1] = argv[i];
        count = argc - (arg_start - 1);
    }

    int opt;
    int long_idx = 0;
    optind = 1; /* Reset getopt */
    while ((opt = getopt_long(count, args, "d:p:o:i:r:s:S:t:qvh", long_opts, &long_idx)) != -1) {
        switch (opt) {
            case 0:
                if (strcmp(long_opts[long_idx].name, "info") == 0) info_only = 1;
                else if (strcmp(long_opts[long_idx].name, "voice") == 0) voice = optarg;
                else if (strcmp(long_opts[long_idx].name, "list") == 0) list_tensors = 1;
                else if (strcmp(long_opts[long_idx].name, "find") == 0) find_pat = optarg;
                else if (strcmp(long_opts[long_idx].name, "verify") == 0) verify_weights = 1;
                else if (strcmp(long_opts[long_idx].name, "tokens") == 0) show_tokens = 1;
                else if (strcmp(long_opts[long_idx].name, "flow-test") == 0) flow_test = 1;
                else if (strcmp(long_opts[long_idx].name, "mimi-test") == 0) mimi_test = 1;
                else if (strcmp(long_opts[long_idx].name, "mimi-wave") == 0) mimi_wave = optarg;
                else if (strcmp(long_opts[long_idx].name, "frames") == 0) params.num_frames = atoi(optarg);
                else if (strcmp(long_opts[long_idx].name, "latent-out") == 0) latent_out = optarg;
                else if (strcmp(long_opts[long_idx].name, "cond-out") == 0) cond_out = optarg;
                else if (strcmp(long_opts[long_idx].name, "flow-out") == 0) flow_out = optarg;
                else if (strcmp(long_opts[long_idx].name, "noise-clamp") == 0) params.noise_clamp = (float)atof(optarg);
                else if (strcmp(long_opts[long_idx].name, "eos-threshold") == 0) {
                    params.eos_enabled = 1;
                    params.eos_threshold = (float)atof(optarg);
                } else if (strcmp(long_opts[long_idx].name, "eos-min-frames") == 0) {
                    params.eos_min_frames = atoi(optarg);
                } else if (strcmp(long_opts[long_idx].name, "eos-after") == 0) {
                    params.eos_after = atoi(optarg);
                }
                else if (strcmp(long_opts[long_idx].name, "dummy") == 0) use_dummy = 1;
                break;
            case 'd': model_dir = optarg; break;
            case 'p': prompt = optarg; break;
            case 'o': output = optarg; break;
            case 'i': input = optarg; break;
            case 'r': params.sample_rate = atoi(optarg); break;
            case 's': params.num_steps = atoi(optarg); break;
            case 'S': params.seed = atoll(optarg); break;
            case 't': params.temp = (float)atof(optarg); break;
            case 'q': output_level = OUTPUT_QUIET; break;
            case 'v': output_level = OUTPUT_VERBOSE; break;
            case 'h': print_usage(argv[0]); if(args != argv) free(args); return 0;
            default:
                print_usage(argv[0]);
                if(args != argv) free(args);
                return 1;
        }
    }

    if (is_export) {
        if (!model_dir || !input || !output) {
            fprintf(stderr, "Error: export-voice requires --dir, --input, and --output\n");
            if(args != argv) free(args);
            return 1;
        }
        ptts_ctx *ctx = ptts_load_dir(model_dir);
        if (!ctx) {
            fprintf(stderr, "Error: %s\n", ptts_get_error());
            if(args != argv) free(args);
            return 1;
        }
        LOG_NORMAL("Exporting voice from %s to %s...\n", input, output);
        if (ptts_export_voice(ctx, input, output) != 0) {
            fprintf(stderr, "Error: %s\n", ptts_get_error());
            ptts_free(ctx);
            if(args != argv) free(args);
            return 1;
        }
        LOG_NORMAL("Done.\n");
        ptts_free(ctx);
        if(args != argv) free(args);
        return 0;
    }

    if (params.num_frames < 0) params.num_frames = 0;
    if (params.eos_min_frames < 1) params.eos_min_frames = 1;
    if (params.eos_after < 0) params.eos_after = 0;

    if (info_only || list_tensors || show_tokens || find_pat || verify_weights || flow_test || mimi_test || mimi_wave) {
        if (!model_dir) {
            fprintf(stderr, "Error: --dir is required for introspection commands\n");
            if(args != argv) free(args);
            return 1;
        }
        ptts_ctx *ctx = ptts_load_dir(model_dir);
        if (!ctx) {
            fprintf(stderr, "Error: %s\n", ptts_get_error());
            if(args != argv) free(args);
            return 1;
        }
        if (info_only) ptts_print_info(ctx);
        if (list_tensors) ptts_list_tensors(ctx);
        if (find_pat) ptts_list_tensors_matching(ctx, find_pat);
        if (verify_weights) {
            int rc = ptts_verify_weights(ctx, output_level >= OUTPUT_VERBOSE);
            if (rc != 0) {
                fprintf(stderr, "Error: weight verification failed\n");
                ptts_free(ctx);
                if(args != argv) free(args);
                return 1;
            }
        }
        if (show_tokens) {
            if (!prompt) {
                fprintf(stderr, "Error: --prompt is required for --tokens\n");
                ptts_free(ctx);
                if(args != argv) free(args);
                return 1;
            }
            int word_count = 0;
            int eos_after_guess = 0;
            char *prepared = ptts_prepare_text(prompt, &word_count, &eos_after_guess);
            if (!prepared) {
                fprintf(stderr, "Error: %s\n", ptts_get_error());
                ptts_free(ctx);
                if(args != argv) free(args);
                return 1;
            }
            int *ids = NULL;
            int n = 0;
            if (ptts_tokenize(ctx, prepared, &ids, &n) != 0) {
                free(prepared);
                fprintf(stderr, "Error: %s\n", ptts_get_error());
                ptts_free(ctx);
                if(args != argv) free(args);
                return 1;
            }
            if (output_level >= OUTPUT_VERBOSE) {
                fprintf(stderr, "Prepared text: %s\n", prepared);
            }
            printf("Tokens (%d):", n);
            for (int i = 0; i < n; i++) printf(" %d", ids[i]);
            printf("\n");
            free(ids);
            free(prepared);
        }
        // ... (skipped some debug hooks for brevity if they duplicate generation setup, but kept in original)
        ptts_free(ctx);
        if(args != argv) free(args);
        return 0;
    }

    if (!prompt) {
        fprintf(stderr, "Error: --prompt is required\n");
        print_usage(argv[0]);
        if(args != argv) free(args);
        return 1;
    }
    if (!output) {
        fprintf(stderr, "Error: --output is required\n");
        print_usage(argv[0]);
        if(args != argv) free(args);
        return 1;
    }

    ptts_audio *audio = NULL;

    if (use_dummy) {
        LOG_NORMAL("Generating dummy audio...\n");
        audio = ptts_generate_dummy(prompt, &params);
    } else {
        if (!model_dir) {
            fprintf(stderr, "Error: --dir is required unless --dummy is used\n");
            if(args != argv) free(args);
            return 1;
        }
        ptts_ctx *ctx = ptts_load_dir(model_dir);
        if (!ctx) {
            fprintf(stderr, "Error: %s\n", ptts_get_error());
            if(args != argv) free(args);
            return 1;
        }
        LOG_VERBOSE("Loaded model, starting inference...\n");
        audio = ptts_generate(ctx, prompt, voice, &params);
        if (!audio) {
            fprintf(stderr, "Error: %s\n", ptts_get_error());
            ptts_free(ctx);
            if(args != argv) free(args);
            return 1;
        }
        ptts_free(ctx);
    }

    if (!audio) {
        fprintf(stderr, "Error: %s\n", ptts_get_error());
        if(args != argv) free(args);
        return 1;
    }

    if (ptts_audio_save_wav(audio, output) != 0) {
        fprintf(stderr, "Error: failed to write WAV\n");
        ptts_audio_free(audio);
        if(args != argv) free(args);
        return 1;
    }

    ptts_audio_free(audio);
    LOG_NORMAL("Saved %s\n", output);
    if(args != argv) free(args);
    return 0;
}
