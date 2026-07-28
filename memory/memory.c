/*
 * memory.c — implementation of neuralc's unified memory interface.
 * All USE_STATIC_MEMORY branching lives here and nowhere else.
 */

#include "memory.h"

#if MEMORY_DEBUG
#include <stdio.h>
#endif

#if USE_STATIC_MEMORY
/* ════════════════════════════════════════════════════════════════
 * STATIC BACKEND — fixed pool, linear allocator, no malloc/free.
 * ════════════════════════════════════════════════════════════════
 * `memory_pool` is a plain static array (.bss, zero-initialized by
 * the loader — not the heap). `offset` is the only piece of state:
 * it only ever moves forward between neuralc_memory_init()/
 * neuralc_memory_reset() calls. No block headers, no free-list, no
 * splitting/coalescing — intentionally, per the "no complex
 * allocators" / "no fragmentation handling" constraints. This trades
 * per-block free() for total predictability, which is what an
 * embedded target with a fixed RAM budget actually wants.
 */

static unsigned char memory_pool[MAX_MEMORY_POOL];
static size_t         offset = 0;

void neuralc_memory_init(void) {
    offset = 0;
#if MEMORY_DEBUG
    printf("[memory] mode=STATIC pool=%d bytes — initialized\n", MAX_MEMORY_POOL);
#endif
}

void neuralc_memory_reset(void) {
    offset = 0;
#if MEMORY_DEBUG
    printf("[memory] static: pool reset (offset -> 0)\n");
#endif
}

void *neuralc_alloc(size_t size) {
    if (size == 0) return NULL;

    /* Overflow-safe bounds check. `offset` is a loop invariant that
     * never exceeds MAX_MEMORY_POOL (every successful alloc keeps it
     * that way, and it only resets to 0), so `MAX_MEMORY_POOL - offset`
     * can never underflow — safe to compare directly against `size`
     * instead of computing `offset + size` (which *could* overflow
     * size_t on a huge bogus `size` before it's ever compared). */
    if (size > (size_t)MAX_MEMORY_POOL - offset) {
#if MEMORY_DEBUG
        printf("[memory] static: FAILED alloc %zu bytes — pool exhausted (%zu/%d used)\n",
               size, offset, MAX_MEMORY_POOL);
#endif
        return NULL;
    }

    void *ptr = &memory_pool[offset];
    offset += size;

#if MEMORY_DEBUG
    printf("[memory] static: alloc %zu bytes  (total usage now %zu/%d)\n",
           size, offset, MAX_MEMORY_POOL);
#endif
    return ptr;
}

void neuralc_free(void *ptr) {
    /* Linear allocator: individual blocks cannot be reclaimed without
     * fragmentation bookkeeping, which is out of scope by design (see
     * memory.h). This is a documented no-op, not a bug — the pool is
     * only ever reclaimed as a whole, via neuralc_memory_reset(). */
    (void)ptr;
#if MEMORY_DEBUG
    if (ptr) printf("[memory] static: neuralc_free(%p) — no-op (linear allocator)\n", ptr);
#endif
}

size_t neuralc_memory_used(void) {
    return offset;
}

#else /* !USE_STATIC_MEMORY */
/* ════════════════════════════════════════════════════════════════
 * DYNAMIC BACKEND — plain malloc/free with a usage counter.
 * ════════════════════════════════════════════════════════════════
 * Behaves exactly like malloc/free would on its own; the counter
 * exists purely so neuralc_memory_used()/MEMORY_DEBUG give the same
 * signal here as they do in the static backend, without needing a
 * heap-walking implementation.
 */
#include <stdlib.h>

static size_t used_bytes = 0;

void neuralc_memory_init(void) {
    used_bytes = 0;
#if MEMORY_DEBUG
    printf("[memory] mode=DYNAMIC (malloc/free) — initialized\n");
#endif
}

void neuralc_memory_reset(void) {
    /* Bookkeeping reset only — does NOT free any outstanding block.
     * Freeing here would double-free pointers the caller may still
     * hold. Every neuralc_alloc() should already be paired with a
     * neuralc_free() by its owner; use neuralc_memory_used() to
     * verify that's true (should read 0) before relying on this. */
    used_bytes = 0;
#if MEMORY_DEBUG
    printf("[memory] dynamic: usage counter reset (bookkeeping only, no frees issued)\n");
#endif
}

void *neuralc_alloc(size_t size) {
    if (size == 0) return NULL;

    void *ptr = malloc(size);
    if (!ptr) {
#if MEMORY_DEBUG
        printf("[memory] dynamic: FAILED alloc %zu bytes — malloc returned NULL\n", size);
#endif
        return NULL;
    }

    used_bytes += size;
#if MEMORY_DEBUG
    printf("[memory] dynamic: alloc %zu bytes  (total usage now %zu)\n", size, used_bytes);
#endif
    return ptr;
}

void neuralc_free(void *ptr) {
    if (!ptr) return;   /* NULL-safe, matches free() semantics explicitly */
#if MEMORY_DEBUG
    /* Log the address *before* freeing — printing `ptr` after free()
     * would read a dangling value the compiler (rightly) flags as a
     * use-after-free, even though only the pointer bits are read. */
    printf("[memory] dynamic: freeing %p\n", ptr);
#endif
    free(ptr);
    /* NOTE: without a header on the block we don't know its exact
     * size here to subtract from used_bytes precisely; callers that
     * need an exact running total in dynamic mode should track sizes
     * at the call site, or switch to MEMORY_DEBUG for per-call sizes.
     * used_bytes therefore reports *cumulative allocated*, not
     * *currently live*, in this minimal dynamic backend. */
}

size_t neuralc_memory_used(void) {
    return used_bytes;
}

#endif /* USE_STATIC_MEMORY */
