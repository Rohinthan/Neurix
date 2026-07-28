/*
 * neurix_cli.c — Standalone Interactive Chatbot Terminal Interface for Neurix v1
 *
 * Provides a rich, auto-discovering REPL shell with styled upper/lower TUI borders
 * for direct communication with trained Neurix models.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/stat.h>

#define MAX_TOKENS      512
#define MAX_NEW_TOKENS  100
#define INPUT_BYTES     4096
#define MAX_VOCAB       65536

// Default hyperparameters
static float g_temperature = 0.80f;
static int   g_top_k       = 20;
static float g_rep_penalty = 1.20f;

// Color and formatting escape codes
#define COLOR_RESET   "\033[0m"
#define COLOR_BOLD    "\033[1m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_WHITE   "\033[37m"

static int file_exists(const char *path) {
    struct stat buffer;
    return (stat(path, &buffer) == 0);
}

// Auto-discover model and vocab files across standard locations
static int auto_discover_files(char *model_out, size_t model_sz,
                               char *vocab_out, size_t vocab_sz) {
    const char *model_candidates[] = {
        "model.bin",
        "/home/raccoon/Neurixv1/model.bin",
        "./assets/model.bin"
    };

    const char *vocab_candidates[] = {
        "assets/vocab.txt",
        "vocab.txt",
        "/home/raccoon/Neurixv1/assets/vocab.txt",
        "./vocab.txt"
    };

    int found_model = 0;
    for (size_t i = 0; i < sizeof(model_candidates)/sizeof(model_candidates[0]); i++) {
        if (file_exists(model_candidates[i])) {
            strncpy(model_out, model_candidates[i], model_sz - 1);
            model_out[model_sz - 1] = '\0';
            found_model = 1;
            break;
        }
    }

    int found_vocab = 0;
    for (size_t i = 0; i < sizeof(vocab_candidates)/sizeof(vocab_candidates[0]); i++) {
        if (file_exists(vocab_candidates[i])) {
            strncpy(vocab_out, vocab_candidates[i], vocab_sz - 1);
            vocab_out[vocab_sz - 1] = '\0';
            found_vocab = 1;
            break;
        }
    }

    return (found_model && found_vocab);
}

static void apply_temperature(float *logits, int vocab, float temp) {
    if (temp < 0.05f) temp = 0.05f;
    for (int i = 0; i < vocab; i++)
        logits[i] /= temp;
}

static void apply_repetition_penalty(float *logits, int vocab, const int *seen, float penalty) {
    for (int i = 0; i < vocab; i++) {
        if (!seen[i]) continue;
        if (logits[i] > 0.0f) logits[i] /= penalty;
        else                  logits[i] *= penalty;
    }
}

static void top_k_filter(float *logits, int vocab, int k) {
    if (k <= 0 || k >= vocab) return;

    static int seen_stamp[MAX_VOCAB];
    static int stamp = 0;
    stamp++;

    for (int pick = 0; pick < k; pick++) {
        float max_val = -1e30f;
        int max_idx = -1;

        for (int i = 0; i < vocab; i++) {
            if (seen_stamp[i] == stamp) continue;
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        if (max_idx >= 0) {
            seen_stamp[max_idx] = stamp;
        }
    }

    for (int i = 0; i < vocab; i++) {
        if (seen_stamp[i] != stamp) {
            logits[i] = -1e30f;
        }
    }
}

static int sample_softmax(const float *logits, int vocab) {
    float max_val = logits[0];
    for (int i = 1; i < vocab; i++)
        if (logits[i] > max_val) max_val = logits[i];

    double sum = 0.0;
    static float probs[MAX_VOCAB];
    for (int i = 0; i < vocab; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }

    double r = ((double)rand() / (double)RAND_MAX) * sum;
    double acc = 0.0;
    for (int i = 0; i < vocab; i++) {
        acc += probs[i];
        if (acc >= r) return i;
    }
    return vocab - 1;
}

static int generate_next_token(const float *raw_logits, int vocab,
                               const int *seen, float temp, int top_k, float rep_pen) {
    static float work_logits[MAX_VOCAB];
    memcpy(work_logits, raw_logits, (size_t)vocab * sizeof(float));

    if (temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab; i++)
            if (work_logits[i] > work_logits[best]) best = i;
        return best;
    }

    apply_repetition_penalty(work_logits, vocab, seen, rep_pen);
    apply_temperature(work_logits, vocab, temp);
    top_k_filter(work_logits, vocab, top_k);

    return sample_softmax(work_logits, vocab);
}

static void run_generation(Pipeline *pipeline, Tokenizer *tokenizer,
                            int vocab_size, int eos_id,
                            const char *prompt_text,
                            int *tokens, float *logits, int *seen) {
    int num_tokens = tokenizer_encode(tokenizer, prompt_text, tokens, MAX_TOKENS);
    if (num_tokens <= 0) {
        printf(COLOR_YELLOW "Neurix > Could not parse prompt input.\n" COLOR_RESET);
        return;
    }

    memset(seen, 0, (size_t)vocab_size * sizeof(int));
    for (int i = 0; i < num_tokens; i++)
        seen[tokens[i]] = 1;

    pipeline_reset(pipeline);
    pipeline_forward(pipeline, tokens, num_tokens, logits);
    const float *cur_logits = logits;

    int max_new_tokens = MAX_NEW_TOKENS;
    if (max_new_tokens > MAX_TOKENS - num_tokens)
        max_new_tokens = MAX_TOKENS - num_tokens;

    // Draw Upper Response Frame Line
    printf(COLOR_GREEN COLOR_BOLD "┌── Neurix AI Response ──────────────────────────────────────┐\n" COLOR_RESET);
    printf(COLOR_GREEN "  " COLOR_RESET);
    fflush(stdout);

    for (int step = 0; step < max_new_tokens; step++) {
        int next_token = generate_next_token(cur_logits, vocab_size, seen,
                                              g_temperature, g_top_k, g_rep_penalty);
        if (next_token < 0 || next_token >= vocab_size) break;

        seen[next_token] = 1;
        tokens[num_tokens++] = next_token;

        char piece[256];
        int written = tokenizer_decode(tokenizer, &next_token, 1, piece, sizeof(piece));
        if (written > 0) {
            if (piece[0] == ' ')
                printf("%s", piece);
            else
                printf(" %s", piece);
            fflush(stdout);
        }

        if (next_token == eos_id) break;

        cur_logits = pipeline_step(pipeline, next_token);
        if (!cur_logits) break;
    }
    
    // Draw Lower Response Frame Line
    printf("\n" COLOR_GREEN COLOR_BOLD "└─────────────────────────────────────────────────────────────┘\n\n" COLOR_RESET);
    fflush(stdout);
}

static void print_banner(const char *model_path, const char *vocab_path) {
    printf(COLOR_CYAN COLOR_BOLD);
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ NEURIX v1 — Interactive Terminal AI Chatbot                 │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Model: %-52s │\n", model_path);
    printf("│ Vocab: %-52s │\n", vocab_path);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Commands: /help  /temp <val>  /topk <val>  /reset  /exit    │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n" COLOR_RESET);
    printf("\n");
}

static void print_help(void) {
    printf(COLOR_CYAN COLOR_BOLD "┌── Neurix Slash Commands ───────────────────────────────────┐\n" COLOR_RESET);
    printf("  /help         - Display command help\n");
    printf("  /temp <val>   - Set sampling temperature (current: %.2f)\n", g_temperature);
    printf("  /topk <val>   - Set Top-K sampling cap (current: %d)\n", g_top_k);
    printf("  /reset        - Reset hidden states & memory\n");
    printf("  /exit         - Quit Neurix CLI\n");
    printf(COLOR_CYAN COLOR_BOLD "└─────────────────────────────────────────────────────────────┘\n\n" COLOR_RESET);
}

int main(int argc, char **argv) {
    srand((unsigned)time(NULL) ^ (unsigned)(uintptr_t)&argc);

    char model_path[1024] = "";
    char vocab_path[1024] = "";

    if (argc >= 3) {
        strncpy(model_path, argv[1], sizeof(model_path) - 1);
        strncpy(vocab_path, argv[2], sizeof(vocab_path) - 1);
    } else {
        if (!auto_discover_files(model_path, sizeof(model_path), vocab_path, sizeof(vocab_path))) {
            fprintf(stderr, COLOR_YELLOW "Error: Model file (model.bin) or Vocab file (assets/vocab.txt) not found.\n" COLOR_RESET);
            fprintf(stderr, "Usage: neurix [model.bin] [assets/vocab.txt]\n");
            return 1;
        }
    }

    Pipeline *pipeline = pipeline_load(model_path);
    if (!pipeline) {
        fprintf(stderr, "Error: Could not load model from '%s'\n", model_path);
        return 1;
    }

    Tokenizer *tokenizer = tokenizer_load(vocab_path);
    if (!tokenizer) {
        fprintf(stderr, "Error: Could not load tokenizer from '%s'\n", vocab_path);
        return 1;
    }

    int vocab_size = tokenizer_vocab_size(tokenizer);
    int eos_id = tokenizer_eos_id(tokenizer);

    int   *tokens = calloc(MAX_TOKENS, sizeof(*tokens));
    float *logits = calloc((size_t)vocab_size, sizeof(*logits));
    int   *seen   = calloc((size_t)vocab_size, sizeof(*seen));

    if (!tokens || !logits || !seen) {
        fprintf(stderr, "Error: Out of memory\n");
        return 1;
    }

    print_banner(model_path, vocab_path);

    char input[INPUT_BYTES];
    while (1) {
        // Upper User Prompt Box Frame
        printf(COLOR_BLUE COLOR_BOLD "┌── User Prompt ──────────────────────────────────────────────┐\n" COLOR_RESET);
        printf(COLOR_BLUE COLOR_BOLD "│ " COLOR_WHITE "User > " COLOR_RESET);
        fflush(stdout);

        if (!fgets(input, sizeof(input), stdin)) break;

        input[strcspn(input, "\r\n")] = '\0';
        if (input[0] == '\0') {
            printf(COLOR_BLUE COLOR_BOLD "└─────────────────────────────────────────────────────────────┘\n\n" COLOR_RESET);
            continue;
        }

        printf(COLOR_BLUE COLOR_BOLD "└─────────────────────────────────────────────────────────────┘\n" COLOR_RESET);

        // Handle slash commands
        if (strcmp(input, "/exit") == 0 || strcmp(input, "/quit") == 0) {
            printf(COLOR_CYAN "\nExiting Neurix Assistant.\n" COLOR_RESET);
            break;
        } else if (strcmp(input, "/help") == 0) {
            print_help();
            continue;
        } else if (strncmp(input, "/temp ", 6) == 0) {
            float t = strtof(input + 6, NULL);
            if (t > 0.0f) {
                g_temperature = t;
                printf(COLOR_GREEN "Temperature updated to %.2f\n\n" COLOR_RESET, g_temperature);
            } else {
                printf(COLOR_YELLOW "Invalid temperature value.\n\n" COLOR_RESET);
            }
            continue;
        } else if (strncmp(input, "/topk ", 6) == 0) {
            int k = atoi(input + 6);
            if (k > 0) {
                g_top_k = k;
                printf(COLOR_GREEN "Top-K updated to %d\n\n" COLOR_RESET, g_top_k);
            } else {
                printf(COLOR_YELLOW "Invalid Top-K value.\n\n" COLOR_RESET);
            }
            continue;
        } else if (strcmp(input, "/reset") == 0) {
            pipeline_reset(pipeline);
            printf(COLOR_GREEN "Model hidden states reset.\n\n" COLOR_RESET);
            continue;
        }

        run_generation(pipeline, tokenizer, vocab_size, eos_id, input, tokens, logits, seen);
    }

    free(tokens);
    free(logits);
    free(seen);
    tokenizer_free(tokenizer);
    return 0;
}
