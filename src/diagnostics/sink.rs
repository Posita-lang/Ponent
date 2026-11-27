//! Diagnostic sinks — the destination of rendered diagnostics.
//!
//! A [`DiagnosticSink`] names *where* rendered text goes without committing
//! the renderer to a byte stream.  The blanket impl keeps every `io::Write`
//! type working unchanged, while dedicated sinks model destinations that are
//! not byte-oriented (LSP channels, wasm callbacks, in-memory strings).
//!
//! [`MakeSink`] creates a *fresh* sink per batch.  The emitter may live for
//! hours, but each `emit_all` gets an exclusive sink, so locking and channel
//! lifetimes stay at the call site.  This is the `tracing-subscriber`
//! `MakeWriter` pattern.
//!
//! ## Emission atomicity
//!
//! Atomicity is a *per-sink* property, not a trait guarantee:
//!
//! * [`StringSink`] is infallible for `write_str`; `write_fmt` surfaces a
//!   formatter error as `Err` like every other sink (a `Display` impl may
//!   legally return `Err(fmt::Error)`), leaving any already-pushed prefix in
//!   the buffer.
//! * A generic `Write` sink (including `stderr`/`BufWriter`) can fail
//!   mid-batch.  For an **unbuffered** sink the bytes written before the error
//!   stay written.  For a **buffered** sink the guarantee holds only if the
//!   emitter calls `flush()` before propagating a mid-batch error *and*
//!   propagates `flush()` errors at batch end.  `BufWriter::drop` does
//!   **attempt** a flush (`let _ = self.flush_buf()`), so buffered bytes are
//!   usually written — but a failure during that drop-time flush is swallowed,
//!   leaving output silently truncated with no error reported.  (An earlier
//!   version of this note said `Drop` *discards* the buffer; that is wrong, and
//!   the distinction matters: the hazard is a silent partial write, not a
//!   silent loss.)  The emitter contract therefore carries the guarantee; this
//!   module only states it.
//! * [`ChannelSink`] sends **one message per `write_str`/`write_fmt` call**,
//!   not per diagnostic: renderers emit many small pieces (header, labels,
//!   gutters), so a receiver sees fragments with no diagnostic boundary
//!   marker.  Coalescing per diagnostic is the emitter's responsibility; until
//!   it exists, a consumer that needs boundaries must frame them itself.
//!   A receiver that drops mid-batch keeps the chunks it already received.
//!   This is the desired LSP/streaming semantics, not a bug.
//!
//! Callers that need all-or-nothing output must render into a
//! [`StringSink`] (or a local `String`) and publish it in one operation.
//!
//! ## Measured allocation counts
//!
//! The claims above are about allocation counts, so they are counted:
//! `cargo bench --bench alloc_counts` wraps `GlobalAlloc` and reports requests per
//! one 64-piece batch (10 B pieces, one warmup batch first, so the pooled rows are
//! steady state).  Recorded on the 4-core WSL2 dev box:
//!
//! | case | allocs / batch |
//! |------|----------------|
//! | `string` — a fresh `StringSink` and nothing else | 7 |
//! | `channel`, 64 × `write_str` | 66 |
//! | `channel`, 8 × `write_parts` of 8 pieces | 9 |
//! | `buffered_channel_with_capacity` | 2 |
//! | `buffered_channel_pooled_with_capacity` | 1 |
//! | pooled `CoalescingSink` over a channel | 1 |
//!
//! Both documented wins are visible in that table: `write_parts` turns per-piece
//! framing into per-group framing (66 → 9), and pooling removes the resident
//! replacement buffer (2 → 1).  The payload allocation survives pooling by design —
//! it *is* the channel message — which is the "what pooling can and cannot save"
//! note on [`BufferedChannelSink`].  The `string` row is not zero because that
//! factory hands out a fresh `String` per batch; it is the growth cost of the
//! 640-byte batch itself, i.e. the floor every other row sits on top of.
//!
//! ## Ordering under reentrancy
//!
//! Every **per-batch buffered** factory — [`stderr_buffered`],
//! [`CoalescingSink`], [`BufferedChannelSink`] — shares one hazard, stated here
//! once instead of per factory: each `make()` returns a *fresh* sink, so a nested
//! emit on the same thread (a `Display` impl that emits another diagnostic, a
//! panic hook) flushes **its** buffer while the outer batch's bytes are still
//! buffered.  The observed order becomes `inner, outer-prefix, outer-suffix`
//! rather than emission order.
//!
//! Only the *unbuffered* sinks are immune: [`stderr`] (one syscall per write) and
//! [`ChannelSink`] (one message per write).  A renderer that can re-enter must use
//! one of those.

use std::fmt;
use std::io::{self, Write};

mod sealed {
    /// Sealing marker: only the sinks in this module (and the blanket
    /// `impl<W: Write>`) may implement [`DiagnosticSink`].
    ///
    /// Without it, the blanket `impl<W: Write>` makes it impossible for ANY
    /// future type that also implements `Write` to provide its own sink impl
    /// (`E0119`), and no new trait method could ship with a sensible default
    /// without breaking external impls.  Types that want to be sinks either
    /// implement `Write` or use a provided sink.
    pub trait Sealed {}
}

/// A destination for rendered diagnostic text.
///
/// Implementors decide what "writing" means: appending to a buffer, sending
/// over a channel, or calling a foreign callback.  The trait is **sealed**:
/// every `io::Write` is already a sink through the blanket impl, and new
/// capabilities can then be added without breaking downstream impls.
///
/// A sink is created by [`MakeSink::make`] and dropped inside a single
/// `Emitter` batch method, so it never crosses a thread boundary and is
/// intentionally *not* required to be `Send`.  This is what lets
/// [`stderr()`] hold an [`io::StderrLock`] for the whole batch:
/// `io::StderrLock` is `!Send`, but it is only ever used on the thread that
/// created it.  Sinks that *are* `Send` (channels, strings) still work; the
/// emitter remains `Send` whenever its renderer and factory are.
pub trait DiagnosticSink: sealed::Sealed {
    /// Write a string slice to the sink.
    fn write_str(&mut self, s: &str) -> io::Result<()>;

    /// Write formatted arguments to the sink.
    ///
    /// Required, not defaulted: the old default built a temporary `String` per
    /// formatted write, which is never the right behaviour for a sink that
    /// cares about allocation — and a future sink that forgot to override it
    /// would pay that silently.
    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> io::Result<()>;

    /// Write several pieces in one logical operation.
    ///
    /// Renderers emit many small pieces per diagnostic; on an **unbuffered**
    /// sink each one is a separate `write(2)`.  The default is the sequential
    /// loop (byte-identical to calling `write_str` N times), so this is purely
    /// additive: a sink that can do better overrides it.
    fn write_parts(&mut self, parts: &[&str]) -> io::Result<()> {
        for p in parts {
            self.write_str(p)?;
        }
        Ok(())
    }

    /// Flush any buffered output.  Defaults to a no-op.
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<W: Write> sealed::Sealed for W {}

// Any `Write` is automatically a sink.
impl<W: Write> DiagnosticSink for W {
    fn write_str(&mut self, s: &str) -> io::Result<()> {
        self.write_all(s.as_bytes())
    }

    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> io::Result<()> {
        // Write directly into the underlying stream; no `String` detour.
        Write::write_fmt(self, args)
    }

    fn write_parts(&mut self, parts: &[&str]) -> io::Result<()> {
        // With one piece there is nothing to vectorise.  `Write::is_write_vectored`
        // is still unstable (`can_vector`, #69941), so the capability query cannot
        // be used here; instead rely on the *documented* contract that the default
        // `write_vectored` writes only the first slice — the advance loop below
        // then makes progress one slice at a time, exactly as the sequential path
        // would, so a sink without real vectored support is correct (never wrong
        // bytes, just no syscall win).
        if parts.len() < 2 {
            for p in parts {
                self.write_all(p.as_bytes())?;
            }
            return Ok(());
        }
        // `UIO_MAXIOV` is 1024 on Linux; std has no stable `write_all_vectored`,
        // so the partial-write loop is explicit: `write_vectored` may consume
        // only a prefix (and some impls only look at the first slice), a short
        // write must advance across slices, and a zero-progress write is an
        // error rather than a spin.
        const MAX_IOV: usize = 1024;
        /// Renderers emit a bounded number of parts per logical unit, so the
        /// common case must not allocate: `IoSlice<'_>` is `Copy`, and `SmallVec`
        /// keeps 16 of them inline.  The heap path only serves a caller that
        /// passes a larger part list in one call.
        const INLINE_PARTS: usize = 16;
        // One buffer for every chunk: the old shape built a fresh `SmallVec` per
        // `MAX_IOV`-part chunk, so a 5 000-part call paid five heap allocations of ~16 KiB for a
        // slice list that is rebuilt each time anyway.
        let mut slices: smallvec::SmallVec<[io::IoSlice<'_>; INLINE_PARTS]> =
            smallvec::SmallVec::new();
        for chunk in parts.chunks(MAX_IOV) {
            slices.clear();
            slices.extend(chunk.iter().map(|p| io::IoSlice::new(p.as_bytes())));
            // An empty slice carries no progress and would make `advance_slices`
            // a no-op; drop them so the loop can never stall on one.
            slices.retain(|s| !s.is_empty());
            let mut pending = slices.as_mut_slice();
            while !pending.is_empty() {
                match self.write_vectored(pending) {
                    Ok(0) => return Err(io::Error::from(io::ErrorKind::WriteZero)),
                    Ok(n) => {
                        // A `Write` that reports more than it was handed would make
                        // `advance_slices` empty `pending` and return success for
                        // bytes nobody wrote.  std's own loops share the exposure;
                        // this is the cheap tripwire (debug-only: the sum is not
                        // worth paying per iteration in release).
                        debug_assert!(
                            n <= pending.iter().map(|s| s.len()).sum::<usize>(),
                            "write_vectored over-reported {n} bytes"
                        );
                        io::IoSlice::advance_slices(&mut pending, n);
                    }
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        Write::flush(self)
    }
}

/// Test sink: collect everything into a `String`.
pub struct StringSink {
    buf: String,
}

impl sealed::Sealed for StringSink {}

impl StringSink {
    pub fn new() -> Self {
        StringSink { buf: String::new() }
    }

    /// Start with `capacity` bytes reserved — one allocation for a
    /// few-KB diagnostic instead of the 2–3 amortized doublings a fresh
    /// `String` pays.
    pub fn with_capacity(capacity: usize) -> Self {
        StringSink {
            buf: String::with_capacity(capacity),
        }
    }

    /// Borrow the accumulated output without consuming the sink.
    pub fn as_str(&self) -> &str {
        &self.buf
    }

    /// Reset the buffer, keeping its capacity for the next batch.
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// Reserved capacity of the backing buffer (for reuse assertions).
    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    /// Consume the sink and return the accumulated output.
    pub fn into_string(self) -> String {
        self.buf
    }
}

impl Default for StringSink {
    fn default() -> Self {
        Self::new()
    }
}

impl DiagnosticSink for StringSink {
    fn write_str(&mut self, s: &str) -> io::Result<()> {
        self.buf.push_str(s);
        Ok(())
    }

    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> io::Result<()> {
        use std::fmt::Write as _;
        // Literal fast path, matching `ChannelSink`/`BufferedChannelSink`/
        // `PooledStringSink`: renderers emit many bare literals (gutters,
        // headers, newlines) and this skips the whole `Formatter` setup.
        if let Some(s) = args.as_str() {
            self.buf.push_str(s);
            return Ok(());
        }
        // Write straight into the internal buffer, avoiding a temporary
        // `String`.
        //
        // This cannot be `unreachable!`: `String` does **not** override
        // `fmt::Write::write_fmt`, so the default implementation propagates
        // `fmt::write`'s result — any `Display` impl in the format chain that
        // returns `Err(fmt::Error)` (entirely legal Rust) reaches here.  Newer
        // std versions also return `Err` on allocation failure instead of
        // aborting.
        //
        // Mapping to `io::Error` keeps the sinks uniform: the blanket
        // `io::Write` path already reports formatter errors that way, and the
        // partial-prefix invariant (bytes before the error stay in the buffer)
        // then holds for every sink instead of "every sink but the one that
        // claimed to be infallible".
        match self.buf.write_fmt(args) {
            Ok(()) => Ok(()),
            // `io::Error::other` is the same shape as `ErrorKind::Other` in
            // `new`, but not deprecated and one call shorter.  `io::const_error!`
            // (message without the `Box`) is still unstable on this toolchain
            // (E0658), and `broken_pipe()`'s payload-free form would drop the
            // message a formatter bug needs.
            Err(_) => Err(io::Error::other("formatter error")),
        }
    }
}

/// Creates a fresh [`DiagnosticSink`] for each batch.
///
/// **Design stance:** the emitter takes a *factory*, not a sink, because a sink
/// carries per-batch state (a buffer, a lock, a pool claim) and reusing one across
/// batches would leak that state into the next batch's output ordering.  A sink
/// whose state is genuinely process-global (a channel sender) should wrap the
/// shared handle in a cheap per-batch value rather than asking the emitter to
/// reuse anything.
///
/// **This trait is sealed by coherence, not by a private supertrait.**  The
/// blanket `impl<F: Fn() -> S>` below means a downstream *type* can never
/// implement it: the compiler must assume the upstream crate may add `Fn` for
/// that type in a future version (E0119, "the upstream crate may add a new impl
/// of trait `Fn` for type …").  In-crate named types are unaffected — coherence
/// sees every local impl — which is why the restriction does not show up in this
/// repository.
///
/// What to do instead: put the configuration in a captured closure, which *is* a
/// `Fn() -> S`.  Every factory here already does exactly that
/// (`buffered_channel_with_capacity`, `stderr_buffered_with`,
/// `pooled_string_with_capacity`), so the pattern is proven rather than
/// theoretical:
///
/// ```ignore
/// // NOT `impl MakeSink for MyFactory` (E0119), but:
/// let make = move || CoalescingSink::new(my_configured_sink(cfg));
/// ```
pub trait MakeSink {
    type Sink: DiagnosticSink;

    /// Produce a new sink.
    fn make(&self) -> Self::Sink;
}

impl<F, S> MakeSink for F
where
    F: Fn() -> S,
    S: DiagnosticSink,
{
    type Sink = S;

    fn make(&self) -> S {
        self()
    }
}

/// Batch small writes for **any** [`DiagnosticSink`].
///
/// [`io::BufWriter`] only helps sinks that are `io::Write`.  A sink that is not —
/// a channel sink, a pooled `String` collector, an LSP transport — had no way to
/// coalesce, so every `write_str`/`write_parts` piece the renderer emitted
/// crossed its boundary (a channel `send`, a mutex acquire, a syscall), i.e. the
/// ~2N-per-batch cost that `stderr_buffered` fixes for stderr only.  This wrapper
/// gives every sink the same coalescing, with the same flush discipline:
/// the emitter flushes at the end of every batch.
///
/// A single piece at least as large as the buffer bypasses it (topping up and
/// flushing immediately would copy it twice), matching `BufWriter`'s own
/// behaviour.
///
/// # Dropping without `flush`
///
/// `Drop` drains best-effort, mirroring `BufWriter` and `BufferedChannelSink`:
/// bytes still in the buffer reach the inner sink instead of being discarded.
/// Without it a panic that emits a diagnostic through this wrapper (a panic hook
/// is the canonical case) lost the entire coalesced prefix, and an early `?`
/// return between two `write_str` calls silently truncated the batch — the
/// "silent partial write with no error reported" hazard this module's header
/// forbids.
///
/// The price is the exposure `BufferedChannelSink`'s ordinary drop path already
/// accepts: `inner.write_str` may block, and during unwinding a blocked sink can
/// hang. That trade is deliberate — losing the bytes is strictly worse, and a
/// generic wrapper cannot tell a blocking inner sink from a non-blocking one. A
/// sink that must never block while unwinding should use
/// [`BufferedChannelSink`], whose panicking drop uses `try_send`.
pub struct CoalescingSink<S: DiagnosticSink> {
    inner: S,
    buf: String,
    capacity: usize,
    /// When set, the buffer is taken from (and returned to) this pool on
    /// creation/drop, so a per-batch sink does not allocate its buffer every
    /// batch.  `None` keeps the plain owning behaviour.
    ///
    /// `&'static` rather than `Arc`: pooling is a process-wide contract (a pool
    /// built per factory call is populated by the sink that dropped into it and
    /// then discarded, so nothing is ever reused), and a `static` is the cheapest
    /// way to express that without an atomic on every take/recycle.
    pool: Option<&'static BufferPool>,
}

impl<S: DiagnosticSink> sealed::Sealed for CoalescingSink<S> {}

impl<S: DiagnosticSink> CoalescingSink<S> {
    /// Wrap `inner` with the default 8 KiB buffer.
    #[must_use]
    pub fn new(inner: S) -> Self {
        Self::with_capacity(inner, 8 * 1024)
    }

    /// Wrap `inner` with a chosen capacity (clamped like `stderr_buffered_with`).
    #[must_use]
    pub fn with_capacity(inner: S, capacity: usize) -> Self {
        CoalescingSink {
            inner,
            buf: String::new(),
            // Same clamping rationale as `stderr_buffered_with`: 0 would make
            // every write flush twice, and an absurd value could abort in
            // `String` growth.
            capacity: clamp_io_capacity(capacity),
            pool: None,
        }
    }

    /// Wrap `inner`, taking the buffer from `pool` (and returning it on drop).
    ///
    /// This is the pooling that `stderr_buffered` lacks: `BufWriter` owns its
    /// `Vec`, so a batch-per-`make()` factory allocated a fresh 16 KiB buffer for
    /// every batch.  The pool degrades to "allocate fresh" when empty, exactly
    /// like [`PooledStringSink`].
    ///
    /// The pool must be `'static` because pooling is a *process-wide* contract:
    /// a pool built per factory call is populated by the sink that dropped into
    /// it and then discarded, so nothing is ever reused.  That is why the
    /// lifetime is part of the signature rather than an incidental detail of
    /// `ArrayQueue`.
    ///
    /// The popped buffer's capacity is unrelated to `capacity`: a smaller one
    /// simply flushes earlier than configured, a larger one is bounded on return
    /// by [`MAX_RECYCLED_CAPACITY`].  Only a `capacity() == 0` shell (an empty
    /// pool) is worth growing, and the drop path refuses to recycle it.
    #[must_use]
    pub fn with_pool(inner: S, capacity: usize, pool: &'static BufferPool) -> Self {
        let mut sink = Self::with_capacity(inner, capacity);
        sink.buf = pool.take(sink.capacity);
        sink.pool = Some(pool);
        sink
    }

    fn drain(&mut self) -> io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        // Clear even on error: the partial bytes were handed to the inner sink (or
        // the write failed wholesale), and keeping them would replay them after
        // the error, which violates the "bytes before the error stay written"
        // invariant.
        let result = self.inner.write_str(&self.buf);
        self.buf.clear();
        result
    }
}

impl<S: DiagnosticSink> Drop for CoalescingSink<S> {
    fn drop(&mut self) {
        // Best-effort drain FIRST, pool return second: a recycled buffer is
        // emptied anyway, so delivering its contents cannot depend on the pool
        // (and the non-pooled case — the default — used to return here and throw
        // the bytes away).  The aliasing with panic-time emission and the
        // blocking trade are argued on the struct's "Dropping without `flush`".
        let _ = self.drain();
        let Some(pool) = self.pool else {
            return;
        };
        // No local guards: `BufferPool::recycle` owns the shell (`capacity() == 0`) and oversized
        // (`capacity() > MAX_RECYCLED_CAPACITY`) rules, and duplicating them here is exactly the
        // drift that centralization was written to end — the only effect was skipping a free
        // `mem::take`.
        let mut buf = std::mem::take(&mut self.buf);
        // `drain` already cleared it on every path; keeping the clear makes the
        // recycled-buffer invariant ("empty, with capacity") independent of that.
        buf.clear();
        // The pool's own guards (shell, oversized) live in `BufferPool::recycle`,
        // so this sink and `PooledStringSink` cannot drift apart again.
        pool.recycle(buf);
    }
}

impl<S: DiagnosticSink> DiagnosticSink for CoalescingSink<S> {
    fn write_str(&mut self, s: &str) -> io::Result<()> {
        if s.len() >= self.capacity {
            self.drain()?;
            return self.inner.write_str(s);
        }
        if self.buf.len() + s.len() > self.capacity {
            self.drain()?;
        }
        self.buf.push_str(s);
        Ok(())
    }

    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> io::Result<()> {
        use fmt::Write as _;
        // Route a bare literal through `write_str`.  Gutters, headers and newlines
        // are the highest-frequency fragments the renderer emits, and this wrapper
        // is what every hot write passes through — yet it was the one sink without
        // the fast path.  Delegating also removes two real asymmetries the old
        // branch had: it drained at `buf.len() >= capacity` *after* formatting (so
        // an exactly-capacity piece flushed via `write_fmt` but not via
        // `write_str`, which tests `len + piece > capacity` first), and it had no
        // oversized bypass (a literal larger than the buffer was formatted into
        // `buf`, growing it past capacity, then copied out — two copies).
        if let Some(s) = args.as_str() {
            return self.write_str(s);
        }
        // Format into the buffer, not into the inner sink: one flush per batch
        // instead of one boundary crossing per formatted fragment.
        if fmt::write(&mut self.buf, args).is_err() {
            // Keep whatever the formatter wrote (the partial-prefix invariant) and
            // report it the way every other sink does.
            return Err(io::Error::other("formatter error"));
        }
        if self.buf.len() >= self.capacity {
            self.drain()?;
        }
        Ok(())
    }

    fn write_parts(&mut self, parts: &[&str]) -> io::Result<()> {
        // Deliberately the sequential path: the inner sink sees one coalesced
        // `write_str`, so vectorising here would only add a branch per call.
        for p in parts {
            self.write_str(p)?;
        }
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.drain()?;
        self.inner.flush()
    }
}

/// Lock stderr once per batch, not once per write.
pub fn stderr() -> impl MakeSink<Sink = io::StderrLock<'static>> {
    || io::stderr().lock()
}

/// [`stderr`] with a pooled 16 KiB coalescing buffer instead of `BufWriter`.
///
/// `BufWriter` allocates its 16 KiB `Vec` in every `make()`, so a per-batch
/// factory pays one allocation per batch forever.  This reuses the buffer through
/// a process-wide [`BufferPool`], and its flush policy is slightly better than
/// `BufWriter`'s for an oversized piece: `CoalescingSink` hands a piece at least
/// as large as the buffer straight to the sink (one `write(2)`), where `BufWriter`
/// fills the remainder of its buffer, flushes, then writes the rest (two).
///
/// The reentrancy caveat is the module-level one ("Ordering under reentrancy").
#[must_use]
pub fn stderr_pooled() -> impl MakeSink<Sink = CoalescingSink<io::StderrLock<'static>>> {
    static POOL: std::sync::LazyLock<BufferPool> =
        std::sync::LazyLock::new(|| BufferPool::new(MAX_POOLED_BUFFERS));
    || CoalescingSink::with_pool(io::stderr().lock(), 16 * 1024, &POOL)
}

/// Like [`stderr`], but batches small writes through a [`BufWriter`].
///
/// `io::stderr()` is unbuffered in Rust: every `write_str` is a `write(2)`
/// syscall.  A batch of N diagnostics therefore costs ~2N syscalls; the
/// buffered factory collapses small writes into 1–2.  The emitter calls
/// `sink.flush()` at the end of every batch, so buffered output is never
/// left behind on the happy path.
///
/// Note that `BufWriter` BYPASSES its buffer for a single `write_all` larger
/// than the remaining capacity (fill, flush, write the remainder directly).
/// A diagnostic whose rendered form exceeds the buffer therefore costs one
/// direct `write(2)` — correct, just not batched; use [`stderr_buffered_with`]
/// to raise the capacity for unusually large renders.
///
/// **Ordering under reentrancy.**  See the module-level "Ordering under
/// reentrancy" section: a fresh `BufWriter` per `make()` flushes a nested emit's
/// bytes ahead of the outer batch's still-buffered prefix.  The unbuffered
/// [`stderr`] does not have this hazard.
#[must_use]
pub fn stderr_buffered() -> impl MakeSink<Sink = io::BufWriter<io::StderrLock<'static>>> {
    stderr_buffered_with(16 * 1024)
}

/// [`stderr_buffered`] with a caller-chosen buffer capacity.
#[must_use]
pub fn stderr_buffered_with(
    capacity: usize,
) -> impl MakeSink<Sink = io::BufWriter<io::StderrLock<'static>>> {
    // `BufWriter::with_capacity(0, …)` is pathological: effectively unbuffered
    // *plus* a flush attempt per write, i.e. strictly worse than plain
    // `stderr()`.  A caller asking for 0 wants "no buffering", which the
    // unbuffered factory already provides; clamping keeps this one a real
    // buffer.
    // Clamped at both ends now: the minimum keeps it a real buffer, the maximum
    // stops one annotated-but-fat-fingered value from aborting the process in
    // `Vec::with_capacity`.
    let capacity = clamp_io_capacity(capacity);
    move || io::BufWriter::with_capacity(capacity, io::stderr().lock())
}

/// Upper bound any sink capacity may request.
///
/// A caller passing `usize::MAX` (or a fat-fingered huge value) previously
/// reached `Vec::with_capacity` and **aborted the process**.  Every
/// capacity-taking entry point clamps against this.
const MAX_SINK_CAPACITY: usize = 16 * 1024 * 1024;

/// Floor for a *syscall-amortising* buffer: `stderr_buffered_with`,
/// `CoalescingSink::with_capacity`.
///
/// Below this the extra `write(2)` per small piece costs more than the memory
/// saved, and `BufWriter::with_capacity(0)` is pathological — effectively
/// unbuffered *plus* a flush attempt per write, strictly worse than plain
/// `stderr()`.
const MIN_IO_BUFFER: usize = 1024;

/// Floor for a *recyclable* buffer: `pooled_string_with_capacity`,
/// `buffered_channel_with_capacity`.
///
/// The pool's only hard rejection is `capacity() == 0` (see
/// `PooledStringSink::drop`); this floor is a *seeding* guideline, not a pool
/// requirement: a buffer smaller than this regrows on the first push, so seeding
/// one is pure waste.  (An earlier comment claimed the pool refuses anything
/// below this — it does not.)
const MIN_POOL_BUFFER: usize = 16;

// Not `const`: `usize::clamp` is `Ord::clamp`, which is `const fn` only under
// the unstable `const_cmp` feature (E0658 on this toolchain). These are called
// once per factory, so the missing `const` costs nothing.
fn clamp_capacity(capacity: usize) -> usize {
    capacity.clamp(MIN_POOL_BUFFER, MAX_SINK_CAPACITY)
}

fn clamp_io_capacity(capacity: usize) -> usize {
    capacity.clamp(MIN_IO_BUFFER, MAX_SINK_CAPACITY)
}

/// Test sink factory: produce a fresh [`StringSink`] per batch.
#[must_use]
pub fn string() -> impl MakeSink<Sink = StringSink> {
    StringSink::new
}

/// Like [`string`], with the batch buffer pre-sized.
///
/// This is the one capacity entry point with **no floor**: a `StringSink` has no
/// pool to poison, so `string_with_capacity(0)` is well-defined and documented —
/// it means "no pre-sizing at all" (a `String::with_capacity(0)` never
/// allocates; the first push grows it).  The ceiling still applies.
#[must_use]
pub fn string_with_capacity(capacity: usize) -> impl MakeSink<Sink = StringSink> {
    let capacity = capacity.min(MAX_SINK_CAPACITY);
    move || StringSink::with_capacity(capacity)
}

/// LSP sink: forward rendered chunks over an mpsc channel.
///
/// Each `write_str` costs exactly one allocation (the owned `String` that goes
/// over the channel) — per-chunk streaming is the intended LSP semantics, not
/// an oversight.  Callers that want fewer allocations and can tolerate larger
/// messages should batch on their side.
///
/// The "format small chunks into a stack buffer" optimization does
/// **not** apply while the payload is `String`: the channel needs an owned value,
/// so `String::from(&stack_buf)` allocates exactly once — the same count as
/// `args.to_string()`, plus a copy.  Removing the allocation requires changing
/// the payload type (`Arc<str>`/`Bytes`, i.e. a copy-free send), which is a wire
/// change for every consumer; until that is decided, this path stays as is.
pub struct ChannelSink {
    tx: crossbeam_channel::Sender<String>,
    /// Set once `send` reports the receiver gone.  A channel cannot come back, so
    /// every later write short-circuits *before* building its payload — the
    /// "keep returning `BrokenPipe`" path is supported (`broken_pipe`'s doc calls
    /// it out), and without this flag each post-disconnect write still paid for a
    /// `String` that was immediately discarded.
    disconnected: bool,
}

/// Payload-free `BrokenPipe`: recurring sends after the receiver dropped must
/// not allocate a fresh `Box<dyn Error>` each time.
#[inline]
fn broken_pipe() -> io::Error {
    io::Error::from(io::ErrorKind::BrokenPipe)
}

impl sealed::Sealed for ChannelSink {}

impl ChannelSink {
    #[inline]
    fn send_owned(&mut self, payload: String) -> io::Result<()> {
        match self.tx.send(payload) {
            Ok(()) => Ok(()),
            Err(_) => {
                self.disconnected = true;
                Err(broken_pipe())
            }
        }
    }
}

impl DiagnosticSink for ChannelSink {
    fn write_str(&mut self, s: &str) -> io::Result<()> {
        if self.disconnected {
            return Err(broken_pipe());
        }
        self.send_owned(s.to_owned())
    }

    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> io::Result<()> {
        if self.disconnected {
            return Err(broken_pipe());
        }
        // A bare literal (`format_args!("…")` with no interpolation) needs no
        // `Formatter` setup at all.
        if let Some(s) = args.as_str() {
            return self.send_owned(s.to_owned());
        }
        // Exactly one allocation: the formatted string sent over the channel.
        self.send_owned(args.to_string())
    }

    fn write_parts(&mut self, parts: &[&str]) -> io::Result<()> {
        if self.disconnected {
            return Err(broken_pipe());
        }
        // One payload, one send: a per-`write_str` frame is explicitly not a
        // contract here (`BufferedChannelSink` already coalesces a whole batch into
        // one message), so the default loop's N allocations and N channel
        // operations buy nothing.  `with_capacity(total)` also makes this the
        // exact-size allocation the single-part path already pays.
        if parts.len() < 2 {
            for p in parts {
                self.write_str(p)?;
            }
            return Ok(());
        }
        let total: usize = parts.iter().map(|p| p.len()).sum();
        // The sequential default performs zero sends for an all-empty part list (`write_all(&[])`
        // is a no-op), so matching it costs one compare and avoids an empty frame at the receiver.
        // Per-`write_str` framing is explicitly not a contract, but "no bytes in, no frame out" is
        // the shape a consumer can reason about.
        if total == 0 {
            return Ok(());
        }
        let mut payload = String::with_capacity(total);
        for p in parts {
            payload.push_str(p);
        }
        self.send_owned(payload)
    }
}

/// LSP sink factory: clone the sender for each batch.
///
/// **Backpressure is the sender's choice, not this factory's.**  `ChannelSink`
/// uses a blocking `send`, so:
///
/// * built from [`crossbeam_channel::unbounded`], a stalled receiver lets every
///   rendered chunk enqueue with nothing pushing back — memory can grow while
///   the session still looks healthy;
/// * built from [`crossbeam_channel::bounded`], `send` blocks once the bound is
///   reached, which is real backpressure: the renderer waits instead of
///   growing.  A wedged receiver then stalls the emitting thread rather than
///   the allocator, and a dropped receiver still surfaces as `BrokenPipe`.
///
/// There is deliberately no wrapper factory that "adds" a bound: the bound
/// belongs to the channel the caller creates, and a factory that silently
/// wrapped a different one would make the actual depth invisible.
/// **Zero-copy payload verdict — recorded so it is not re-attempted:**
/// switching the payload to `Arc<str>`/`Bytes` does *not* remove a copy here.  The
/// renderer produces bytes incrementally, so the batch buffer must be a growable
/// `String`; wrapping it in an `Arc<str>` copies it exactly once either way (the
/// `String` is sent by move today, and `Arc::from(String)`/`Bytes::from(String)`
/// both allocate + copy).  A genuinely zero-copy payload would need the renderer
/// to emit into a fixed-capacity shared buffer it hands over whole — a different
/// architecture, not a type change.
pub fn channel(tx: crossbeam_channel::Sender<String>) -> impl MakeSink<Sink = ChannelSink> {
    move || ChannelSink {
        tx: tx.clone(),
        disconnected: false,
    }
}

/// Batches a whole diagnostic into **one** channel message.
///
/// [`ChannelSink`] sends one message per `write_str`, so a renderer emitting
/// header/labels/gutters produces dozens of allocations and channel operations
/// per diagnostic.  The emitter already scopes a sink to one batch and calls
/// `flush()` at the end, which is exactly the coalescing boundary: this sink
/// appends into a reused buffer and sends once at `flush`.
///
/// Trade-off, stated because it is a real behaviour change: the receiver sees
/// **nothing** until `flush`.  A consumer that needs incremental chunks (a
/// live-progress UI) must keep using [`channel`].  A panic mid-batch would
/// otherwise lose the buffered text, so `Drop` flushes best-effort (ignoring the
/// error, like `BufWriter`'s drop).
///
/// # Memory bound
///
/// The buffer grows to the size of one batch and is then sent as one message:
/// **the batch size is the memory ceiling**, and this sink has none of its own
/// (unlike [`CoalescingSink`], which flushes at a fixed capacity).  A
/// pathological batch therefore builds one arbitrarily large `String` and then
/// one arbitrarily large channel message.  If that must be bounded, wrap this
/// sink in [`CoalescingSink`] — coalescing composes: the receiver then gets one
/// message per capacity window instead of one per batch.
///
/// # What pooling can and cannot save
///
/// Each flushed payload owns its buffer (it is the channel message), so
/// `buffered_channel_pooled_with_capacity` recycles only the sink's *resident*
/// replacement buffer: one allocation per batch, not per flush.  Removing the
/// per-flush allocation would need a copy-free payload (`Arc<str>`/`Bytes`),
/// which the zero-copy verdict above already rules out.
///
/// **Precondition: one `flush` per `make`/drop cycle.**  Every flush takes a fresh resident buffer
/// from the pool and hands the old one to the channel, which owns it and never returns it; only
/// `Drop` recycles one.  That balances at the emitter's documented "flush at the end of each
/// batch" contract.  A sink flushed more often than it is made (a future per-diagnostic flush, a
/// user-driven mid-batch flush) drains the pool after `MAX_POOLED_BUFFERS` flushes and then
/// allocates on every flush while `pool.len()` reads 0 — i.e. the pooled variant degrades to the
/// unpooled one plus pool churn.  Stated here so a flush-frequency change is forced to revisit
/// this claim instead of silently invalidating it.
pub struct BufferedChannelSink {
    tx: crossbeam_channel::Sender<String>,
    buf: String,
    /// Capacity every replacement buffer is seeded with (see `send_buffered`).
    capacity: usize,
    /// When set, the resident replacement buffer comes from this pool and is
    /// returned to it on drop.  Only the *resident* buffer can be recycled — see
    /// the struct's "What pooling can and cannot save".
    pool: Option<&'static BufferPool>,
    disconnected: bool,
}

impl BufferedChannelSink {
    /// The buffer that replaces `buf` after a flush.
    ///
    /// Pooled sinks take it from the pool (one allocation saved per batch);
    /// unpooled ones allocate at the configured capacity, which is why
    /// `mem::take` — a capacity-0 shell — was the wrong primitive here.
    fn fresh_buffer(&self) -> String {
        match self.pool {
            Some(pool) => pool.take(self.capacity),
            None => String::with_capacity(self.capacity),
        }
    }

    /// Frame and send the buffered text, clearing it for the next batch.
    fn send_buffered(&mut self) -> io::Result<()> {
        // Sticky, like `write_str`/`write_fmt`: once the channel is gone every
        // method keeps reporting it.  Landing on `Ok(())` here made a flush retry
        // at batch end look successful on a dead channel — the "silently
        // succeeding" failure mode this module's docs forbid.
        if self.disconnected {
            return Err(broken_pipe());
        }
        if self.buf.is_empty() {
            return Ok(());
        }
        // Hand the payload on and keep the configured capacity.  `mem::take` left
        // a capacity-0 shell, so the *second* diagnostic of a batch (and all
        // later ones) regrew through the doubling chain — exactly the cost
        // `buffered_channel_with_capacity` exists to avoid.
        // Two steps, not one: `mem::replace(&mut self.buf, self.fresh_buffer())`
        // borrows `self` both mutably and immutably (E0502) — `fresh_buffer`
        // reads `self.pool`/`self.capacity` while `self.buf` is mutably borrowed.
        let replacement = self.fresh_buffer();
        let payload = std::mem::replace(&mut self.buf, replacement);
        match self.tx.send(payload) {
            Ok(()) => Ok(()),
            Err(_) => {
                // `send` returned the payload in the error; drop it with the
                // error rather than keeping a stale buffer around.
                self.disconnected = true;
                Err(broken_pipe())
            }
        }
    }

    /// Non-blocking variant, used from `Drop` while unwinding.
    ///
    /// A blocking `send` there can hang forever: with a `bounded` channel the
    /// receiver being stalled is *why* it is full, and the panicking thread is
    /// inside a destructor — the receiver may be waiting on it.  Losing the
    /// buffered tail is strictly better than a deadlock.
    fn try_flush(&mut self) -> io::Result<()> {
        if self.disconnected {
            return Err(broken_pipe());
        }
        if self.buf.is_empty() {
            return Ok(());
        }
        // Two steps, not one: `mem::replace(&mut self.buf, self.fresh_buffer())`
        // borrows `self` both mutably and immutably (E0502) — `fresh_buffer`
        // reads `self.pool`/`self.capacity` while `self.buf` is mutably borrowed.
        let replacement = self.fresh_buffer();
        let payload = std::mem::replace(&mut self.buf, replacement);
        match self.tx.try_send(payload) {
            Ok(()) => Ok(()),
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                self.disconnected = true;
                Err(broken_pipe())
            }
            // Full: drop the tail (documented above).
            Err(crossbeam_channel::TrySendError::Full(_)) => Ok(()),
        }
    }
}

impl Drop for BufferedChannelSink {
    fn drop(&mut self) {
        // Best-effort: a dropped batch still reaches the receiver, and a
        // disconnected one has nowhere to go.  The error is swallowed for the
        // same reason `BufWriter::drop` swallows its flush error — there is no
        // caller left to report to.
        //
        // While unwinding the send must not block (see `try_flush`); the
        // documented backpressure semantics stay for ordinary drops.
        let _ = if std::thread::panicking() {
            self.try_flush()
        } else {
            self.send_buffered()
        };
        // The payload that was flushed owns its buffer; what comes back to the pool
        // is the resident replacement buffer (see "What pooling can and cannot
        // save").  Unpooled sinks have nothing to do.
        if let Some(pool) = self.pool {
            pool.recycle(std::mem::take(&mut self.buf));
        }
    }
}

impl sealed::Sealed for BufferedChannelSink {}

impl DiagnosticSink for BufferedChannelSink {
    fn write_str(&mut self, s: &str) -> io::Result<()> {
        if self.disconnected {
            return Err(broken_pipe());
        }
        self.buf.push_str(s);
        Ok(())
    }

    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> io::Result<()> {
        use std::fmt::Write as _;
        if self.disconnected {
            return Err(broken_pipe());
        }
        if let Some(s) = args.as_str() {
            self.buf.push_str(s);
            return Ok(());
        }
        match self.buf.write_fmt(args) {
            Ok(()) => Ok(()),
            Err(_) => Err(io::Error::other("formatter error")),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        self.send_buffered()
    }
}

/// [`channel`] with one message per batch instead of per write.
#[must_use]
pub fn buffered_channel(
    tx: crossbeam_channel::Sender<String>,
) -> impl MakeSink<Sink = BufferedChannelSink> {
    buffered_channel_with_capacity(tx, 4096)
}

/// [`buffered_channel`] with an explicit initial buffer size.
#[must_use]
pub fn buffered_channel_with_capacity(
    tx: crossbeam_channel::Sender<String>,
    capacity: usize,
) -> impl MakeSink<Sink = BufferedChannelSink> {
    let capacity = clamp_capacity(capacity);
    move || BufferedChannelSink {
        tx: tx.clone(),
        buf: String::with_capacity(capacity),
        capacity,
        pool: None,
        disconnected: false,
    }
}

/// [`buffered_channel_with_capacity`] whose resident buffer comes from `pool`.
///
/// Pooling saves **one allocation per batch**, not one per flush: each flushed
/// payload owns its buffer because it *is* the channel message.  See the sink's
/// "What pooling can and cannot save".
#[must_use]
pub fn buffered_channel_pooled_with_capacity(
    tx: crossbeam_channel::Sender<String>,
    capacity: usize,
    pool: &'static BufferPool,
) -> impl MakeSink<Sink = BufferedChannelSink> {
    let capacity = clamp_capacity(capacity);
    move || BufferedChannelSink {
        tx: tx.clone(),
        buf: pool.take(capacity),
        capacity,
        pool: Some(pool),
        disconnected: false,
    }
}

/// Retained buffers for [`pooled_string`].  Bounded: a process that renders one
/// huge batch and then many small ones must not pin the huge allocation forever.
const MAX_POOLED_BUFFERS: usize = 8;
/// Capacity ceiling for a recycled buffer.
///
/// `MAX_POOLED_BUFFERS` bounds the *count*; without a byte bound a single 1 MB
/// batch buffer stays resident for the pool's lifetime and is handed to later small batches on
/// the queue's round-robin — pinned for good and charged to each of them, exactly the "must not
/// pin the huge allocation" failure the pool's doc claims to prevent.
///
/// (This said the buffer "entered the (LIFO) pool and was popped first".  `crossbeam_queue::
/// ArrayQueue` is a bounded **FIFO queue**: `push` appends at the tail and `pop` takes the head,
/// so `take` returns the *oldest* recycled buffer.  The bound is required under either
/// discipline; only the warmth story differs, and LIFO would need either a lock (the lock-free
/// queue is what lets the parallel emitter share one pool) or unsafe tagging.)
const MAX_RECYCLED_CAPACITY: usize = 64 * 1024;

/// Pop-with-fallback and guarded recycle, shared by every pooled sink.
///
/// The policy used to be copy-pasted per sink and had already drifted:
/// `PooledStringSink::drop` cleared the buffer and rejected shells and oversized
/// allocations, `CoalescingSink::drop` did none of the three — so a flush-only
/// batch could fill a bounded pool with capacity-0 shells and every later batch
/// re-grew from zero while `pool.len()` looked healthy.  One type owns the rules
/// now; a new pooled sink gets them by construction.
///
/// All operations are lock-free (`crossbeam_queue::ArrayQueue`), so a pool can be
/// shared across the parallel emitter's worker threads.  A full or empty pool is
/// never an error: `recycle` drops the buffer, `take` allocates.
pub struct BufferPool {
    /// Bounded MPMC **FIFO queue** (not a stack — `ArrayQueue` appends at the tail and pops from
    /// the head, so `take` returns the oldest recycled buffer).  `push` fails when full, `pop`
    /// fails when empty → allocate fresh; a lock-free queue means a pool can be shared by the
    /// parallel emitter's worker threads with no contention on the fast path.
    ///
    /// Eager rather than `LazyLock`: `LazyLock<T>`'s `F` defaults to `fn() -> T`,
    /// so a *capturing* closure (the `slots` parameter) would not coerce — and a
    /// pool is constructed once per factory, not on a hot path.
    queue: crossbeam_queue::ArrayQueue<String>,
}

impl BufferPool {
    /// A pool holding at most `slots` idle buffers.
    ///
    /// `slots` bounds the *count*; [`MAX_RECYCLED_CAPACITY`] bounds the bytes of
    /// each recycled buffer.
    ///
    /// `slots` is clamped to at least 1: `ArrayQueue::new(0)` is legal (every push fails, every
    /// take allocates), but a zero-slot pool is always a caller mistake and the clamp keeps
    /// `pool.len()` meaningful for the residency assertions.  Stated rather than implicit —
    /// the same reason [`MIN_POOL_BUFFER`]'s doc needed a correction.
    #[must_use]
    pub fn new(slots: usize) -> Self {
        BufferPool {
            queue: crossbeam_queue::ArrayQueue::new(slots.max(1)),
        }
    }

    /// Pop a cleared buffer, allocating one with `capacity` when the pool is
    /// empty.
    ///
    /// The popped buffer's capacity is unrelated to `capacity`: a smaller one
    /// simply means the caller's buffer flushes earlier than configured, a larger
    /// one is bounded on return by [`MAX_RECYCLED_CAPACITY`].  Only seeding should
    /// assume they match.
    #[must_use]
    pub fn take(&self, capacity: usize) -> String {
        let mut buf = self
            .queue
            .pop()
            .unwrap_or_else(|| String::with_capacity(capacity));
        // A recycled buffer is already empty; the clear keeps the invariant
        // independent of the recycle path.
        buf.clear();
        buf
    }

    /// Return `buf` for reuse, unless recycling it would poison the pool.
    ///
    /// Two rejections, both of them real bugs before they were centralised:
    ///
    /// * `capacity() == 0` — a shell (e.g. after `into_string`), which would
    ///   "reuse" nothing and force the next batch to re-grow from zero;
    /// * `capacity() > MAX_RECYCLED_CAPACITY` — a runaway render must not pin a
    ///   huge allocation for the pool's lifetime, where LIFO would hand it to
    ///   every later small batch.
    ///
    /// The buffer is cleared first so a recycled buffer never carries one batch's
    /// bytes into the next.
    pub fn recycle(&self, mut buf: String) {
        buf.clear();
        if buf.capacity() == 0 || buf.capacity() > MAX_RECYCLED_CAPACITY {
            return;
        }
        // Fails (and drops `buf`) when the pool is full — the queue's capacity is
        // the count bound, so there is no separate length check to drift.
        let _ = self.queue.push(buf);
    }

    /// Number of idle buffers (for reuse assertions).
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether the pool currently holds no idle buffer.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Count only: dumping the buffers would print every batch's bytes.
        f.debug_struct("BufferPool")
            .field("idle", &self.len())
            .finish_non_exhaustive()
    }
}

// `Debug` impls that report sizes, not contents.  Without them `{:?}` on a sink
// would not compile, which tempts a future `#[derive(Debug)]` on a sink holding a
// multi-KiB batch buffer — a derive that dumps kilobytes into a panic message.
// Each of these prints the shape a reader actually wants.

impl std::fmt::Debug for StringSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StringSink")
            .field("len", &self.buf.len())
            .field("capacity", &self.buf.capacity())
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for ChannelSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelSink")
            .field("disconnected", &self.disconnected)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for BufferedChannelSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferedChannelSink")
            .field("len", &self.buf.len())
            .field("capacity", &self.capacity)
            .field("disconnected", &self.disconnected)
            .finish_non_exhaustive()
    }
}

impl<S: DiagnosticSink> std::fmt::Debug for CoalescingSink<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The inner sink is deliberately not formatted: it may itself be a
        // buffered sink, and nesting the dump is how kilobytes leak into a panic
        // message.
        f.debug_struct("CoalescingSink")
            .field("len", &self.buf.len())
            .field("capacity", &self.capacity)
            .field("pooled", &self.pool.is_some())
            .finish_non_exhaustive()
    }
}

/// [`StringSink`] whose buffer is returned to a shared pool on drop.
///
/// The emitter makes one sink per batch, so a long-lived emitter paid one
/// allocation (plus 2–3 amortized doublings) per batch forever.  The pool is
/// reused instead; it aims for one buffer per concurrent batch, kept across
/// batches.
///
/// The pool is `Arc<Mutex<Vec<String>>>` rather than a thread-local so a sink can
/// be moved between worker threads (the parallel emitter does exactly that).  A
/// poisoned or contended pool degrades to "allocate a fresh buffer", never to an
/// error.
pub struct PooledStringSink {
    buf: String,
    /// Bounded MPMC stack, shared by every pooled sink.  Its capacity is
    /// `MAX_POOLED_BUFFERS` and the recycle guards live in [`BufferPool`], so
    /// `CoalescingSink::with_pool` and this sink cannot drift apart again.
    pool: std::sync::Arc<BufferPool>,
}

impl PooledStringSink {
    /// Borrow the accumulated output without consuming the sink.
    pub fn as_str(&self) -> &str {
        &self.buf
    }

    /// Consume the sink and return the accumulated output.
    ///
    /// The returned `String` owns the allocation, so this **forfeits** the pooled
    /// buffer for one batch (`Drop` then sees a capacity-0 shell and does not
    /// pollute the pool with it).  Batch consumers should keep reading
    /// [`Self::as_str`] and let `drop` recycle — that is the path the emitter
    /// takes.
    pub fn into_string(mut self) -> String {
        std::mem::take(&mut self.buf)
    }

    /// Reserved capacity of the backing buffer (for reuse assertions).
    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }
}

impl Drop for PooledStringSink {
    fn drop(&mut self) {
        // `recycle` owns all three guards (shell, oversized, clear) — this sink
        // used to carry its own copy, which is exactly how `CoalescingSink` came
        // to be missing them.
        self.pool.recycle(std::mem::take(&mut self.buf));
    }
}

impl fmt::Debug for PooledStringSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Length, not contents: a diagnostic buffer dumps kilobytes into a
        // `{:?}` for no benefit.
        f.debug_struct("PooledStringSink")
            .field("len", &self.buf.len())
            .field("capacity", &self.buf.capacity())
            .finish()
    }
}

impl sealed::Sealed for PooledStringSink {}

impl DiagnosticSink for PooledStringSink {
    fn write_str(&mut self, s: &str) -> io::Result<()> {
        self.buf.push_str(s);
        Ok(())
    }

    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> io::Result<()> {
        use std::fmt::Write as _;
        if let Some(s) = args.as_str() {
            self.buf.push_str(s);
            return Ok(());
        }
        match self.buf.write_fmt(args) {
            Ok(()) => Ok(()),
            Err(_) => Err(io::Error::other("formatter error")),
        }
    }
}

/// [`string`] backed by a shared, bounded buffer pool.
#[must_use]
pub fn pooled_string() -> impl MakeSink<Sink = PooledStringSink> {
    pooled_string_with_capacity(4096)
}

/// [`pooled_string`] with an explicit initial buffer size (used only on the first
/// batch, before the pool warms up).
#[must_use]
pub fn pooled_string_with_capacity(capacity: usize) -> impl MakeSink<Sink = PooledStringSink> {
    let capacity = clamp_capacity(capacity);
    let pool = std::sync::Arc::new(BufferPool::new(MAX_POOLED_BUFFERS));
    // Seed one buffer so the very first batch does not allocate; `recycle` applies
    // the same guards as every later return, so a mis-seeded pool cannot happen.
    pool.recycle(String::with_capacity(capacity));
    move || PooledStringSink {
        buf: pool.take(capacity),
        pool: std::sync::Arc::clone(&pool),
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod coalescing_probe_tests {
    use super::*;

    /// Counts boundary crossings (the thing buffering exists to reduce) without
    /// timing: a shared counter behind an `io::Write`.
    struct Counting(std::sync::Arc<std::sync::atomic::AtomicUsize>);

    impl io::Write for Counting {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn coalescing_cuts_boundary_crossings_but_not_bytes() {
        let pieces: Vec<String> = (0..64).map(|i| format!("line {i}\n")).collect();
        let plain_counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut plain = Counting(std::sync::Arc::clone(&plain_counter));
        let mut plain_bytes = String::new();
        for p in &pieces {
            io::Write::write_all(&mut plain, p.as_bytes()).expect("write");
            plain_bytes.push_str(p);
        }
        io::Write::flush(&mut plain).expect("flush");
        let plain_crossings = plain_counter.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(plain_crossings, pieces.len(), "one crossing per piece");

        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut coalesced =
            CoalescingSink::with_capacity(Counting(std::sync::Arc::clone(&counter)), 1024);
        let mut coalesced_bytes = String::new();
        for p in &pieces {
            coalesced.write_str(p).expect("write");
            coalesced_bytes.push_str(p);
        }
        coalesced.flush().expect("flush");
        assert_eq!(
            coalesced_bytes, plain_bytes,
            "coalescing must not change the bytes"
        );
        assert_eq!(
            counter.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "64 short pieces fit one 1 KiB buffer"
        );
    }

    #[test]
    fn coalescing_bypasses_the_buffer_for_an_oversized_piece() {
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut sink =
            CoalescingSink::with_capacity(Counting(std::sync::Arc::clone(&counter)), 1024);
        let huge = "x".repeat(4096);
        sink.write_str("small").expect("write");
        sink.write_str(&huge).expect("write");
        sink.flush().expect("flush");
        // small → buffered, huge → direct, flush → the buffered prefix: two
        // crossings, because the oversized piece never entered the buffer.
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    /// Records what actually reached the sink, so "was this delivered?" is
    /// observable *after* the wrapper has been dropped.
    struct Shared(std::sync::Arc<std::sync::Mutex<String>>);

    impl io::Write for Shared {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .expect("shared sink poisoned")
                .push_str(std::str::from_utf8(buf).expect("utf-8 pieces only"));
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn coalescing_pool_returns_the_buffer_on_drop() {
        // `ArrayQueue::new` is not a const fn, so the pool lives behind a
        // `LazyLock` (the production pools do the same).
        static POOL: std::sync::LazyLock<BufferPool> =
            std::sync::LazyLock::new(|| BufferPool::new(4));
        // Seed a *real* buffer: the old version popped an empty pool, wrote
        // nothing, and asserted that the resulting capacity-0 shell came back —
        // enshrining the bug that filled a bounded pool with shells that reuse
        // nothing.
        POOL.recycle(String::with_capacity(1024));
        {
            let mut sink = CoalescingSink::with_pool(Vec::<u8>::new(), 1024, &POOL);
            sink.write_str("x").expect("write");
        }
        assert_eq!(POOL.len(), 1, "the buffer must be recycled on drop");
        let recycled = POOL.take(0);
        assert_eq!(recycled.capacity(), 1024, "the allocation is reused");
        assert!(recycled.is_empty(), "a recycled buffer is always empty");
    }

    /// The shell case, which the test above used to *be*.
    ///
    /// `with_pool` no longer *hands out* shells — `BufferPool::take` allocates at
    /// the requested capacity when the pool is empty — so the guard is asserted
    /// where it now lives: `recycle` refuses a capacity-0 buffer.  `into_string`
    /// (which leaves exactly such a shell behind) is the production path that
    /// depends on it.
    #[test]
    fn buffer_pool_refuses_to_recycle_a_shell() {
        let pool = BufferPool::new(4);
        pool.recycle(String::new());
        assert_eq!(pool.len(), 0, "a capacity-0 shell must not be recycled");
        // A cold `take` allocates at the requested capacity, so the pool can never
        // be *repopulated* with a shell through this path either.
        let cold = pool.take(1024);
        assert_eq!(cold.capacity(), 1024, "cold take pre-sizes");
        assert!(cold.is_empty());
    }

    /// The oversized guard, on the same centralised path: a runaway render must
    /// not pin a huge allocation in a bounded pool.
    #[test]
    fn buffer_pool_refuses_to_recycle_an_oversized_buffer() {
        let pool = BufferPool::new(4);
        pool.recycle(String::with_capacity(MAX_RECYCLED_CAPACITY + 1));
        assert_eq!(pool.len(), 0, "oversized buffers are dropped, not pooled");
        pool.recycle(String::with_capacity(MAX_RECYCLED_CAPACITY));
        assert_eq!(pool.len(), 1, "the bound itself is recyclable");
    }

    /// The measurable claim behind pooling, made checkable without a counting
    /// allocator: the second batch writes into the *same* allocation as the first.
    /// (`as_str().as_ptr()` is the buffer's data pointer; a re-allocation would
    /// move it.)  The module's allocation arguments were unmeasured before this.
    #[test]
    fn pooled_batches_reuse_the_same_allocation() {
        let make = pooled_string_with_capacity(1024);
        let first = {
            let mut sink = make.make();
            sink.write_str("batch one").expect("write");
            sink.as_str().as_ptr()
        };
        let second = {
            let mut sink = make.make();
            assert_eq!(sink.capacity(), 1024, "the pooled buffer came back");
            sink.write_str("batch two").expect("write");
            sink.as_str().as_ptr()
        };
        assert_eq!(
            first, second,
            "no allocation between batches: the pool handed back the same buffer"
        );
    }

    /// ...and the coalescing wrapper gets the same guarantee through `with_pool`.
    #[test]
    fn coalescing_pooled_sink_reuses_the_buffer() {
        static POOL: std::sync::LazyLock<BufferPool> =
            std::sync::LazyLock::new(|| BufferPool::new(4));
        let drained = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let first_ptr = {
            let mut sink =
                CoalescingSink::with_pool(Shared(std::sync::Arc::clone(&drained)), 1024, &POOL);
            sink.write_str("first").expect("write");
            let p = sink.buf.as_ptr().cast::<u8>();
            drop(sink);
            p
        };
        let second_ptr = {
            let mut sink =
                CoalescingSink::with_pool(Shared(std::sync::Arc::clone(&drained)), 1024, &POOL);
            sink.write_str("second").expect("write");
            sink.buf.as_ptr().cast::<u8>()
        };
        assert_eq!(
            first_ptr, second_ptr,
            "the coalescing pool handed back the same allocation"
        );
    }

    /// Dropping without an explicit `flush` must still deliver the coalesced
    /// prefix. `BufWriter` and `BufferedChannelSink` both do this; the wrapper
    /// used to discard the bytes entirely, which is how a panic hook emitting a
    /// diagnostic lost its whole message.
    #[test]
    fn coalescing_drop_delivers_buffered_bytes() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        {
            let mut sink = CoalescingSink::new(Shared(std::sync::Arc::clone(&seen)));
            sink.write_str("first ").expect("write");
            sink.write_fmt(format_args!("second {}", 2)).expect("fmt");
            // No `flush()`: the drop is the point of the test.
        }
        assert_eq!(seen.lock().unwrap().as_str(), "first second 2");
    }

    /// ...and the oversized-write bypass path is covered too, because it hands the
    /// piece straight to the inner sink: a `Drop` that only drained the buffer
    /// would look correct here while losing the rest.
    #[test]
    fn coalescing_drop_delivers_after_an_oversized_bypass() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let big = "y".repeat(2048);
        {
            let mut sink =
                CoalescingSink::with_capacity(Shared(std::sync::Arc::clone(&seen)), 1024);
            sink.write_str("head").expect("write");
            sink.write_str(&big).expect("oversized");
            sink.write_str("tail").expect("write");
        }
        let got = seen.lock().unwrap();
        assert_eq!(got.len(), 4 + big.len() + 4);
        assert!(got.starts_with("head"));
        assert!(got.ends_with("tail"));
    }

    /// The `write_fmt` twin of `string_sink_write_fmt_keeps_the_prefix_and_errors`:
    /// a `Display` impl that fails *after* writing must leave the prefix in the
    /// buffer and surface the error (the coalescing path had no such test, so a
    /// future `write_fmt` rewrite could silently drop the prefix).
    #[test]
    fn coalescing_write_fmt_keeps_the_prefix_and_errors() {
        struct HalfBroken;
        impl fmt::Display for HalfBroken {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("prefix")?;
                Err(fmt::Error)
            }
        }

        let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let mut sink = CoalescingSink::new(Shared(std::sync::Arc::clone(&seen)));
        let err = sink
            .write_fmt(format_args!("{}", HalfBroken))
            .expect_err("a formatter error must surface as io::Error");
        assert_eq!(err.kind(), io::ErrorKind::Other);
        // The prefix is still *in the buffer* (not yet handed to the inner sink),
        // and the drop delivers it — both halves of the partial-prefix invariant.
        drop(sink);
        assert_eq!(seen.lock().unwrap().as_str(), "prefix");
    }

    /// A bare literal and the same text through `write_str` must be byte- and
    /// boundary-identical: `write_fmt` now delegates for literals, so the
    /// drain-at-threshold rule cannot differ between the two entry points.
    #[test]
    fn coalescing_write_fmt_literal_matches_write_str() {
        let cross = |write_str: bool| {
            let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let bytes = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
            let mut sink = CoalescingSink::with_capacity(
                CountingAndRecording(
                    std::sync::Arc::clone(&counter),
                    std::sync::Arc::clone(&bytes),
                ),
                1024,
            );
            for _ in 0..600 {
                if write_str {
                    sink.write_str("literal").expect("write");
                } else {
                    sink.write_fmt(format_args!("literal")).expect("fmt");
                }
            }
            sink.flush().expect("flush");
            (
                counter.load(std::sync::atomic::Ordering::Relaxed),
                bytes.lock().unwrap().clone(),
            )
        };
        let via_str = cross(true);
        let via_fmt = cross(false);
        assert_eq!(via_str.1, via_fmt.1, "same bytes");
        assert_eq!(via_str.0, via_fmt.0, "same boundary-crossing count");
    }

    /// Counts crossings *and* records the payload, for the parity test above.
    struct CountingAndRecording(
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
        std::sync::Arc<std::sync::Mutex<String>>,
    );

    impl io::Write for CountingAndRecording {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.1
                .lock()
                .expect("recorder poisoned")
                .push_str(std::str::from_utf8(buf).expect("utf-8 pieces only"));
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    /// One batch, one message — and a batch that is dropped without an explicit
    /// `flush` still reaches the receiver (the panic-containment path).
    /// `write_parts` must be byte-identical to the sequential default even when
    /// the sink reports vectored support but only ever consumes a *prefix* of the
    /// iovecs — the case std's (still unstable) `write_all_vectored` exists for,
    /// and the one a hand-rolled loop gets wrong.
    #[test]
    fn write_parts_matches_sequential_on_partial_writes() {
        use std::io::{IoSlice, Write};

        struct Trickle {
            buf: Vec<u8>,
            per_call: usize,
            /// `false` models a sink with the *default* `write_vectored` (first
            /// slice only), which is what a console/`Write` without real support
            /// looks like.
            vectored: bool,
        }
        impl Write for Trickle {
            fn write(&mut self, b: &[u8]) -> io::Result<usize> {
                let n = b.len().min(self.per_call.max(1));
                self.buf.extend_from_slice(&b[..n]);
                Ok(n)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
            fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
                if !self.vectored {
                    // std's default: only the first slice is considered.
                    return match bufs.first() {
                        Some(b) => self.write(b),
                        None => Ok(0),
                    };
                }
                let mut written = 0;
                for b in bufs {
                    if written >= self.per_call {
                        break;
                    }
                    let n = b.len().min(self.per_call - written);
                    self.buf.extend_from_slice(&b[..n]);
                    written += n;
                }
                Ok(written)
            }
        }

        // Empty pieces and multi-byte UTF-8 are the interesting cases: an empty
        // `IoSlice` would stall `advance_slices`, and `γ` straddles slice
        // boundaries in the per-byte trickle.
        let parts = ["head\n", "", "  \u{3b3}  label\n", "tail\n", ""];
        let mut expected = Vec::new();
        for p in parts {
            expected.extend_from_slice(p.as_bytes());
        }
        for per_call in [1usize, 2, 3, 7, 64, usize::MAX] {
            for vectored in [false, true] {
                let mut sink = Trickle {
                    buf: Vec::new(),
                    per_call,
                    vectored,
                };
                sink.write_parts(&parts).unwrap();
                assert_eq!(
                    sink.buf, expected,
                    "per_call={per_call} vectored={vectored}"
                );
            }
        }

        // A sink that accepts nothing must surface `WriteZero`, not spin.
        struct Stuck;
        impl Write for Stuck {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Ok(0)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
            fn write_vectored(&mut self, _: &[IoSlice<'_>]) -> io::Result<usize> {
                Ok(0)
            }
        }
        let mut stuck = Stuck;
        assert_eq!(
            stuck.write_parts(&["a", "b"]).unwrap_err().kind(),
            io::ErrorKind::WriteZero
        );
    }

    #[test]
    fn buffered_channel_sends_one_message_per_batch() {
        use crate::diagnostics::sink::MakeSink as _;
        let (tx, rx) = crossbeam_channel::unbounded();
        let make = buffered_channel(tx);
        {
            let mut sink = make.make();
            for chunk in ["head\n", "  ^^^^ label\n", "tail\n"] {
                sink.write_str(chunk).unwrap();
            }
            assert!(rx.try_recv().is_err(), "nothing is sent before flush");
            sink.flush().unwrap();
        }
        let msg = rx.try_recv().expect("one message per flushed batch");
        assert_eq!(msg, "head\n  ^^^^ label\ntail\n");
        assert!(rx.try_recv().is_err(), "exactly one message, not three");

        // Drop without flush: best-effort delivery — the receiver must be *alive*
        // for this to test anything.  The old block dropped the receiver
        // immediately (`unbounded().0`), so the sink was disconnected from birth,
        // the drop-flush failed silently, and no assertion covered the path the
        // comment claims.
        {
            let (tx_drop, rx_drop) = crossbeam_channel::unbounded();
            let mut sink = buffered_channel_with_capacity(tx_drop, 16).make();
            sink.write_str("dropped").unwrap();
            drop(sink);
            assert_eq!(
                rx_drop
                    .try_recv()
                    .expect("drop must flush the buffered tail"),
                "dropped"
            );
        }

        // A disconnected receiver short-circuits without allocating.
        let (tx2, rx2) = crossbeam_channel::unbounded();
        let mut sink = buffered_channel(tx2).make();
        drop(rx2);
        sink.write_str("x").unwrap();
        assert_eq!(sink.flush().unwrap_err().kind(), io::ErrorKind::BrokenPipe);
        assert_eq!(
            sink.write_str("y").unwrap_err().kind(),
            io::ErrorKind::BrokenPipe
        );
    }

    /// The pooled factory must hand back the *same* allocation across batches
    /// once warm; that is the whole point of the pool.
    #[test]
    fn pooled_string_reuses_its_buffer_across_batches() {
        let make = pooled_string_with_capacity(4096);
        for _ in 0..4 {
            let mut sink = make.make();
            sink.write_str("diagnostic body").unwrap();
            assert_eq!(sink.as_str(), "diagnostic body");
            // The batch path (`as_str` + drop) must recycle the allocation, so
            // the first 4096-byte buffer is never lost.
            assert!(
                sink.capacity() >= 4096,
                "capacity {} lost from the pool",
                sink.capacity()
            );
        }
        // `into_string` hands the allocation to the caller by design; the pool
        // must then fall back to a fresh buffer rather than reusing a 0-capacity
        // shell.
        let mut sink = make.make();
        sink.write_str("taken").unwrap();
        assert_eq!(sink.into_string(), "taken");
        let mut sink = make.make();
        sink.write_str("x").unwrap();
        assert_eq!(sink.as_str(), "x");

        // Pool must stay bounded.
        let make = pooled_string();
        let sinks: Vec<_> = (0..MAX_POOLED_BUFFERS * 3).map(|_| make.make()).collect();
        drop(sinks);
    }

    #[test]
    fn string_sink_accumulates() {
        let mut sink = StringSink::new();
        sink.write_str("hello ").unwrap();
        sink.write_fmt(format_args!("{}", 42)).unwrap();
        assert_eq!(sink.into_string(), "hello 42");
    }

    #[test]
    fn write_types_are_sinks() {
        // The blanket impl makes any `Write` a sink (the trait itself is
        // intentionally not `Send`-bound; see the module docs).
        fn assert_sink<S: DiagnosticSink>(_: &S) {}
        let buf: Vec<u8> = Vec::new();
        assert_sink(&buf);
    }

    #[test]
    fn channel_sink_forwards_chunks() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let make = channel(tx);
        let mut sink = make.make();
        sink.write_str("a").unwrap();
        sink.write_str("b").unwrap();
        assert_eq!(rx.recv().unwrap(), "a");
        assert_eq!(rx.recv().unwrap(), "b");
    }

    /// `ChannelSink::write_parts` coalesces into one message: N-1 fewer
    /// allocations and N-1 fewer channel operations for a multi-part emit.  The
    /// framing change is sanctioned by `BufferedChannelSink`, which already sends
    /// one message per batch — per-`write_str` framing is not a contract.
    #[test]
    fn channel_sink_write_parts_sends_one_message() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let make = channel(tx);
        let mut sink = make.make();
        sink.write_parts(&["head", " ^^^^", "tail"]).expect("parts");
        assert_eq!(rx.recv().unwrap(), "head ^^^^tail");
        assert!(
            rx.try_recv().is_err(),
            "multi-part writes must not fragment the message"
        );
        // The single-part path keeps `write_str`'s semantics.
        sink.write_parts(&["solo"]).expect("one part");
        assert_eq!(rx.recv().unwrap(), "solo");
    }

    /// Pooling the buffered channel sink saves the *resident* buffer's allocation
    /// per batch — the flushed payload owns its buffer by design, so this is the
    /// one allocation pooling can remove.  Pointer identity is the proof: a second
    /// batch reuses the same allocation.
    #[test]
    fn buffered_channel_pooled_reuses_the_resident_buffer() {
        static POOL: std::sync::LazyLock<BufferPool> =
            std::sync::LazyLock::new(|| BufferPool::new(4));
        let (tx, rx) = crossbeam_channel::unbounded();
        let make = buffered_channel_pooled_with_capacity(tx, 1024, &POOL);
        let first = {
            let mut sink = make.make();
            sink.write_str("first").expect("write");
            // The flush hands the *payload* buffer over and swaps in the resident
            // one; that replacement is what the pool sees at drop.
            sink.flush().expect("flush");
            let resident = sink.buf.as_ptr();
            assert_eq!(rx.recv().expect("payload"), "first");
            drop(sink);
            resident
        };
        assert_eq!(POOL.len(), 1, "the resident buffer was recycled");
        let mut second = make.make();
        assert_eq!(
            second.buf.as_ptr(),
            first,
            "the next batch reused that allocation"
        );
        second.write_str("second").expect("write");
        second.flush().expect("flush");
        assert_eq!(rx.recv().expect("payload"), "second");
    }

    /// `BufWriter<W>` and `W` produce identical bytes for the same write
    /// sequence — buffering only changes syscall granularity, not content.
    /// This is the property `stderr_buffered` relies on.
    #[test]
    fn buffered_writer_preserves_byte_content() {
        // The claim under test is "`DiagnosticSink::flush` forwards to
        // `Write::flush`".  The old version called `Write::flush` *first* (and
        // `into_inner` also flushes), so a `DiagnosticSink::flush` that was a
        // no-op would still pass.  Now the sink's flush is the ONLY flush, and a
        // counting wrapper proves it reached the underlying writer.
        struct Counting<W> {
            inner: W,
            flushes: usize,
        }
        impl<W: Write> Write for Counting<W> {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.inner.write(buf)
            }
            fn flush(&mut self) -> io::Result<()> {
                self.flushes += 1;
                self.inner.flush()
            }
        }

        let mut source = Counting {
            inner: Vec::new(),
            flushes: 0,
        };
        {
            let mut buffered = io::BufWriter::with_capacity(16, &mut source);
            for chunk in ["hello ", "world\n", "line 2\n", "x"] {
                buffered.write_all(chunk.as_bytes()).unwrap();
            }
            // Disambiguate the two `flush` methods; the emitter calls this one.
            DiagnosticSink::flush(&mut buffered).unwrap();
            assert!(
                buffered.get_ref().flushes >= 1,
                "DiagnosticSink::flush must forward to the inner writer"
            );
            assert_eq!(
                buffered.buffer().len(),
                0,
                "a forwarded flush must drain the BufWriter"
            );
        }
        assert_eq!(source.inner, b"hello world\nline 2\nx");
    }

    /// A `Display` impl may legally fail *after* writing a prefix; the documented
    /// contract is that the prefix stays in the buffer and the error surfaces as
    /// `io::Error`.  Both halves are pinned here — the old suite had no test for
    /// either, so a refactor to `format!` + `push_str` would have violated the
    /// contract silently.
    #[test]
    fn string_sink_write_fmt_keeps_the_prefix_and_errors() {
        struct HalfBroken;
        impl fmt::Display for HalfBroken {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("prefix")?;
                Err(fmt::Error)
            }
        }

        let mut sink = StringSink::new();
        let err = sink
            .write_fmt(format_args!("{}", HalfBroken))
            .expect_err("a formatter error must surface as io::Error");
        assert_eq!(err.kind(), io::ErrorKind::Other);
        assert_eq!(
            sink.as_str(),
            "prefix",
            "the bytes written before the failure must stay in the buffer"
        );
    }

    /// A single `write_all` larger than the buffer capacity takes the
    /// documented bypass path (fill, flush, write the remainder directly) and
    /// must still preserve every byte.
    #[test]
    fn buffered_writer_bypasses_oversized_write_without_loss() {
        let mut direct = Vec::new();
        let mut buffered = io::BufWriter::with_capacity(8, Vec::new());
        let big = "0123456789abcdefghijklmnopqrstuvwxyz";
        direct.write_all(big.as_bytes()).unwrap();
        buffered.write_all(big.as_bytes()).unwrap();
        Write::flush(&mut buffered).unwrap();
        assert_eq!(direct, buffered.into_inner().unwrap());
    }

    /// Every sink must produce identical bytes for the same interleaving of
    /// `write_str` / `write_fmt`; a `write_fmt` implementation that drifts from
    /// its `write_str` partner is the class of bug this pins.
    #[test]
    fn sinks_agree_on_byte_content() {
        let mut string_sink = StringSink::new();
        let mut direct: Vec<u8> = Vec::new();
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut channel_sink = ChannelSink {
            tx,
            disconnected: false,
        };
        let mut channel_out = String::new();

        for (i, chunk) in ["a", "beta\n", "γ"].iter().enumerate() {
            string_sink.write_str(chunk).unwrap();
            DiagnosticSink::write_str(&mut direct, chunk).unwrap();
            channel_sink.write_str(chunk).unwrap();
            channel_out.push_str(&rx.recv().unwrap());
            string_sink.write_fmt(format_args!("[{i}]")).unwrap();
            DiagnosticSink::write_fmt(&mut direct, format_args!("[{i}]")).unwrap();
            channel_sink.write_fmt(format_args!("[{i}]")).unwrap();
            channel_out.push_str(&rx.recv().unwrap());
        }

        let expected = string_sink.into_string();
        assert_eq!(String::from_utf8(direct).unwrap(), expected);
        assert_eq!(channel_out, expected);
    }

    /// A receiver that drops mid-batch yields `BrokenPipe` and keeps the
    /// chunks already delivered; later writes keep returning `Err` rather than
    /// panicking or silently succeeding.
    #[test]
    fn channel_sink_reports_broken_pipe_after_receiver_drop() {
        let (tx, rx) = crossbeam_channel::unbounded();
        let mut sink = ChannelSink {
            tx,
            disconnected: false,
        };
        sink.write_str("kept").unwrap();
        drop(rx);
        let err = sink.write_str("lost").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::BrokenPipe);
        assert_eq!(
            sink.write_fmt(format_args!("x")).unwrap_err().kind(),
            io::ErrorKind::BrokenPipe
        );
    }

    #[test]
    fn string_sink_reuses_capacity() {
        let mut sink = StringSink::with_capacity(4096);
        sink.write_str("abc").unwrap();
        assert_eq!(sink.as_str(), "abc");
        sink.clear();
        assert_eq!(sink.as_str(), "");
        assert!(sink.capacity() >= 4096);
    }
}
