//! Allocation counts per sink batch — the measurement behind the sink docs.
//!
//! `src/diagnostics/sink.rs` argues in prose that buffering cuts boundary
//! crossings, that `ChannelSink::write_parts` is one allocation instead of N, and
//! that pooling saves the per-batch buffer.  Those are claims about allocation
//! counts, so they are *counted* here instead of asserted: a [`GlobalAlloc`]
//! wrapper tallies allocation requests, each case runs one representative batch,
//! and the table prints what it cost.
//!
//! The root crate is a binary (empty lib target), so this bench `#[path]`-includes
//! the sink module — the same trick `ascii_dispatch` uses.
//!
//! ```text
//! cargo bench --bench alloc_counts
//! ```
//!
//! Deliberately a `harness = false` binary and **not** a unit test: a
//! `#[global_allocator]` in the test binary would perturb every other test's
//! contention and timing, and allocation counts cannot be attributed per test
//! under `--test-threads`.
//!
//! Reading the table: a "batch" is 64 pieces of 10 bytes (`write_str`), which is
//! the shape the emitter produces for one diagnostic (header, gutter, labels).
//! Every case is measured **after** one warmup batch, so a case reports its steady
//! state — for the pooled sinks that is exactly the question ("what does the
//! second batch cost?"), and for the unpooled ones it changes nothing.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

/// Counts allocation *requests*; the deallocation path is untouched so the
/// measurement cannot change allocator behaviour.
struct Counting;

// SAFETY: every method forwards to `System` unchanged. The counter is a relaxed
// atomic increment with no allocation in it, so it cannot recurse into the
// allocator. `alloc`, `alloc_zeroed` and `realloc` are the only ways to obtain
// memory and all three are counted, so a case's delta is exactly the number of
// allocations it performed.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[path = "../src/diagnostics/sink.rs"]
#[allow(dead_code)]
mod sink;

use sink::{BufferPool, CoalescingSink, DiagnosticSink, MakeSink};

const PIECES: usize = 64;
const PIECE: &str = "label:0000";

fn pieces() -> Vec<String> {
    (0..PIECES)
        .map(|i| format!("{}{i:02}", &PIECE[..PIECE.len() - 2]))
        .collect()
}

/// Allocation requests performed by `f`.
fn allocs<F: FnOnce()>(f: F) -> usize {
    let before = ALLOCS.load(Ordering::Relaxed);
    f();
    ALLOCS.load(Ordering::Relaxed) - before
}

/// One batch: every piece through `write_str`, then a flush, then drop — the
/// whole lifecycle the emitter gives a sink, so a `Drop` that drains (and
/// therefore allocates, for a channel sink) is included rather than hidden.
fn batch<S: DiagnosticSink>(mut s: S, pieces: &[String]) {
    for p in pieces {
        s.write_str(p).expect("write");
    }
    s.flush().expect("flush");
}

/// Same batch, but handing the pieces to `write_parts` in groups of 8.
fn batch_parts<S: DiagnosticSink>(mut s: S, pieces: &[String]) {
    let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
    for chunk in refs.chunks(8) {
        s.write_parts(chunk).expect("parts");
    }
    s.flush().expect("flush");
}

fn drain(rx: &crossbeam_channel::Receiver<String>) {
    while rx.try_recv().is_ok() {}
}

fn main() {
    let pieces = pieces();
    // Built before every measured scope, so the piece `String`s are not attributed
    // to the sinks.
    let mut rows: Vec<(&str, usize)> = Vec::new();

    // The floor: an in-memory sink the renderer writes into. Nothing is allocated
    // after the sink's own buffer is warm.
    let make = sink::string();
    batch(make.make(), &pieces);
    rows.push((
        "string sink (floor: renderer writes only)",
        allocs(|| batch(make.make(), &pieces)),
    ));

    // ChannelSink: one owned payload per `write_str`, per-write framing.
    let (tx, rx) = crossbeam_channel::unbounded();
    let make = sink::channel(tx);
    batch(make.make(), &pieces);
    drain(&rx);
    rows.push((
        "channel, write_str x64",
        allocs(|| batch(make.make(), &pieces)),
    ));
    drain(&rx);

    // ...and through `write_parts`: 8 payloads instead of 64.
    let (tx, rx) = crossbeam_channel::unbounded();
    let make = sink::channel(tx);
    batch_parts(make.make(), &pieces);
    drain(&rx);
    rows.push((
        "channel, write_parts in groups of 8",
        allocs(|| batch_parts(make.make(), &pieces)),
    ));
    drain(&rx);

    // BufferedChannelSink: one message per batch, but a fresh replacement buffer
    // per flush — the cost the pooled variant exists to remove.
    let (tx, rx) = crossbeam_channel::unbounded();
    let make = sink::buffered_channel_with_capacity(tx, 4096);
    batch(make.make(), &pieces);
    drain(&rx);
    rows.push((
        "buffered_channel, unpooled",
        allocs(|| batch(make.make(), &pieces)),
    ));
    drain(&rx);

    static POOL: std::sync::LazyLock<BufferPool> = std::sync::LazyLock::new(|| BufferPool::new(8));
    let (tx, rx) = crossbeam_channel::unbounded();
    let make = sink::buffered_channel_pooled_with_capacity(tx, 4096, &POOL);
    batch(make.make(), &pieces); // warms the pool
    drain(&rx);
    rows.push((
        "buffered_channel, pooled (buffer recycled)",
        allocs(|| batch(make.make(), &pieces)),
    ));
    drain(&rx);

    // CoalescingSink over an unbuffered channel, pooled: one send per capacity
    // window instead of one per piece, and no buffer allocation per batch.
    static CPOOL: std::sync::LazyLock<BufferPool> = std::sync::LazyLock::new(|| BufferPool::new(8));
    let (tx, rx) = crossbeam_channel::unbounded();
    let factory = sink::channel(tx);
    batch(
        CoalescingSink::with_pool(factory.make(), 4096, &CPOOL),
        &pieces,
    );
    drain(&rx);
    rows.push((
        "coalescing(pooled) over channel",
        allocs(|| {
            batch(
                CoalescingSink::with_pool(factory.make(), 4096, &CPOOL),
                &pieces,
            )
        }),
    ));
    drain(&rx);

    println!(
        "\nallocations per batch ({PIECES} pieces x {} B)\n",
        PIECE.len()
    );
    println!("{:<44} {:>6}", "case", "allocs");
    println!("{:<44} {:->6}", "", "");
    for (name, n) in &rows {
        println!("{name:<44} {n:>6}");
    }
    println!(
        "\n`string` is the floor, and it is not zero: `sink::string()` builds a fresh\n\
         `StringSink` per batch, so the row is that buffer's own growth over 640 B.\n\
         `channel, write_str` is one payload allocation per piece; `write_parts` cuts\n\
         the same bytes to one per group (8 here), and the pooled sinks remove the\n\
         resident replacement buffer — never the payload, which owns its allocation\n\
         because it *is* the channel message.\n"
    );
}
