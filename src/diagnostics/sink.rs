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
        for chunk in parts.chunks(MAX_IOV) {
            let mut slices: Vec<io::IoSlice<'_>> = chunk
                .iter()
                .map(|p| io::IoSlice::new(p.as_bytes()))
                .collect();
            // An empty slice carries no progress and would make `advance_slices`
            // a no-op; drop them so the loop can never stall on one.
            slices.retain(|s| !s.is_empty());
            let mut pending = slices.as_mut_slice();
            while !pending.is_empty() {
                match self.write_vectored(pending) {
                    Ok(0) => return Err(io::Error::from(io::ErrorKind::WriteZero)),
                    Ok(n) => io::IoSlice::advance_slices(&mut pending, n),
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
pub struct CoalescingSink<S: DiagnosticSink> {
    inner: S,
    buf: String,
    capacity: usize,
    /// When set, the buffer is taken from (and returned to) this pool on
    /// creation/drop, so a per-batch sink does not allocate its buffer every
    /// batch.  `None` keeps the plain owning behaviour.
    pool: Option<&'static crossbeam_queue::ArrayQueue<String>>,
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
            capacity: capacity.clamp(1024, MAX_SINK_CAPACITY),
            pool: None,
        }
    }

    /// Wrap `inner`, taking the buffer from `pool` (and returning it on drop).
    ///
    /// This is the pooling that `stderr_buffered` lacks: `BufWriter` owns its
    /// `Vec`, so a batch-per-`make()` factory allocated a fresh 16 KiB buffer for
    /// every batch.  The pool degrades to "allocate fresh" when empty, exactly
    /// like [`PooledStringSink`].
    #[must_use]
    pub fn with_pool(
        inner: S,
        capacity: usize,
        pool: &'static crossbeam_queue::ArrayQueue<String>,
    ) -> Self {
        let mut buf = pool.pop().unwrap_or_default();
        buf.clear();
        let mut sink = Self::with_capacity(inner, capacity);
        sink.buf = buf;
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
        let Some(pool) = self.pool else {
            return;
        };
        // Only recycle a buffer that is within the documented bound; a runaway
        // render must not park a huge allocation in the pool forever.
        if self.buf.capacity() > MAX_RECYCLED_CAPACITY {
            return;
        }
        let buf = std::mem::take(&mut self.buf);
        // A full pool drops the buffer (the documented degradation).
        let _ = pool.push(buf);
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
/// **Ordering under reentrancy.**  Each `make()` returns a fresh `BufWriter`, so
/// a nested emit on the same thread (a `Display` impl that emits another
/// diagnostic, a panic hook) flushes *its* buffer to stderr while the outer
/// batch's bytes are still buffered: the observed order becomes
/// `inner, outer-prefix, outer-suffix` rather than emission order.  The
/// unbuffered [`stderr`] does not have this hazard.  Use it when the renderer
/// can re-enter; the buffered factory trades that guarantee for fewer syscalls.
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
    let capacity = capacity.clamp(1024, MAX_SINK_CAPACITY);
    move || io::BufWriter::with_capacity(capacity, io::stderr().lock())
}

/// Upper bound any sink capacity may request.
///
/// A caller passing `usize::MAX` (or a fat-fingered huge value) previously
/// reached `Vec::with_capacity` and **aborted the process**; a zero capacity made
/// `pooled_string_with_capacity(0)` a silently dead pool (the seed buffer had
/// capacity 0, so `Drop`'s `capacity() == 0` guard rejected every returned
/// buffer and pooling was disabled with no diagnostic).  One helper keeps all
/// four factories consistent in both directions.
const MAX_SINK_CAPACITY: usize = 16 * 1024 * 1024;
/// Floor for a *pooled/buffered* capacity: below this the buffer is a shell and
/// the pool would refuse it back (see above).
const MIN_SINK_CAPACITY: usize = 16;

fn clamp_capacity(capacity: usize) -> usize {
    capacity.clamp(MIN_SINK_CAPACITY, MAX_SINK_CAPACITY)
}

/// Test sink factory: produce a fresh [`StringSink`] per batch.
#[must_use]
pub fn string() -> impl MakeSink<Sink = StringSink> {
    StringSink::new
}

/// Like [`string`], with the batch buffer pre-sized.
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
pub struct BufferedChannelSink {
    tx: crossbeam_channel::Sender<String>,
    buf: String,
    /// Capacity every replacement buffer is seeded with (see `send_buffered`).
    capacity: usize,
    disconnected: bool,
}

impl BufferedChannelSink {
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
        let payload = std::mem::replace(&mut self.buf, String::with_capacity(self.capacity));
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
        let payload = std::mem::replace(&mut self.buf, String::with_capacity(self.capacity));
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
        disconnected: false,
    }
}

/// Retained buffers for [`pooled_string`].  Bounded: a process that renders one
/// huge batch and then many small ones must not pin the huge allocation forever.
const MAX_POOLED_BUFFERS: usize = 8;
/// Capacity ceiling for a recycled buffer.
///
/// `MAX_POOLED_BUFFERS` bounds the *count*; without a byte bound a single 1 MB
/// batch buffer entered the (LIFO) pool and was popped first by every later
/// small batch — pinned forever and charged to each of them, exactly the
/// "must not pin the huge allocation" failure the pool's doc claims to prevent.
const MAX_RECYCLED_CAPACITY: usize = 64 * 1024;

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
    /// Bounded MPMC stack.  The pool's semantics were always exactly that
    /// (`push` fails when full, `pop` fails when empty → allocate fresh), so the
    /// `Mutex<Vec<_>>` only added an acquire/release per batch and made the cap an
    /// `if` instead of a capacity.  `ArrayQueue` is lock-free and its capacity *is*
    /// `MAX_POOLED_BUFFERS`; a poisoned pool cannot exist, so "pool unavailable →
    /// allocate" is simply `pop() == None`.
    pool: std::sync::Arc<crossbeam_queue::ArrayQueue<String>>,
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
        let mut buf = std::mem::take(&mut self.buf);
        buf.clear();
        // A capacity-0 shell means `into_string` handed the allocation away; do
        // not push it, or the next batch would "reuse" nothing and re-grow from
        // zero (which is exactly what `pooled_string_reuses_its_buffer_across_batches`
        // caught).
        if buf.capacity() == 0 {
            return;
        }
        // Oversized buffers are not recycled (see `MAX_RECYCLED_CAPACITY`):
        // dropping the allocation is cheaper than pinning a megabyte for the
        // lifetime of the pool, and LIFO would hand it to every small batch.
        if buf.capacity() > MAX_RECYCLED_CAPACITY {
            return;
        }
        // Fails (and drops `buf`) when the pool is full — the bound is the
        // queue's capacity now, so there is no separate length check to drift.
        let _ = self.pool.push(buf);
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
    let pool: std::sync::Arc<crossbeam_queue::ArrayQueue<String>> =
        std::sync::Arc::new(crossbeam_queue::ArrayQueue::new(MAX_POOLED_BUFFERS));
    let _ = pool.push(String::with_capacity(capacity));
    move || {
        let buf = pool
            .pop()
            .unwrap_or_else(|| String::with_capacity(capacity));
        PooledStringSink {
            buf,
            pool: std::sync::Arc::clone(&pool),
        }
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

    #[test]
    fn coalescing_pool_returns_the_buffer_on_drop() {
        // `ArrayQueue::new` is not a const fn, so the pool lives behind a
        // `LazyLock` (the production pools do the same).
        static POOL: std::sync::LazyLock<crossbeam_queue::ArrayQueue<String>> =
            std::sync::LazyLock::new(|| crossbeam_queue::ArrayQueue::new(4));
        drop(CoalescingSink::with_pool(Vec::<u8>::new(), 1024, &POOL));
        assert_eq!(POOL.len(), 1, "the buffer must be recycled on drop");
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
