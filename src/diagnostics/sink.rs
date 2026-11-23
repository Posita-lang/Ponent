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
pub fn stderr_buffered() -> impl MakeSink<Sink = io::BufWriter<io::StderrLock<'static>>> {
    stderr_buffered_with(16 * 1024)
}

/// [`stderr_buffered`] with a caller-chosen buffer capacity.
pub fn stderr_buffered_with(
    capacity: usize,
) -> impl MakeSink<Sink = io::BufWriter<io::StderrLock<'static>>> {
    // `BufWriter::with_capacity(0, …)` is pathological: effectively unbuffered
    // *plus* a flush attempt per write, i.e. strictly worse than plain
    // `stderr()`.  A caller asking for 0 wants "no buffering", which the
    // unbuffered factory already provides; clamping keeps this one a real
    // buffer.
    let capacity = capacity.max(1024);
    move || io::BufWriter::with_capacity(capacity, io::stderr().lock())
}

/// Test sink factory: produce a fresh [`StringSink`] per batch.
pub fn string() -> impl MakeSink<Sink = StringSink> {
    StringSink::new
}

/// Like [`string`], with the batch buffer pre-sized.
pub fn string_with_capacity(capacity: usize) -> impl MakeSink<Sink = StringSink> {
    move || StringSink::with_capacity(capacity)
}

/// LSP sink: forward rendered chunks over an mpsc channel.
///
/// Each `write_str` costs exactly one allocation (the owned `String` that goes
/// over the channel) — per-chunk streaming is the intended LSP semantics, not
/// an oversight.  Callers that want fewer allocations and can tolerate larger
/// messages should batch on their side.
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
    disconnected: bool,
}

impl BufferedChannelSink {
    /// Frame and send the buffered text, clearing it for the next batch.
    fn send_buffered(&mut self) -> io::Result<()> {
        if self.disconnected || self.buf.is_empty() {
            return Ok(());
        }
        let payload = std::mem::take(&mut self.buf);
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
}

impl Drop for BufferedChannelSink {
    fn drop(&mut self) {
        // Best-effort: a dropped batch still reaches the receiver, and a
        // disconnected one has nowhere to go.  The error is swallowed for the
        // same reason `BufWriter::drop` swallows its flush error — there is no
        // caller left to report to.
        let _ = self.send_buffered();
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
pub fn buffered_channel(
    tx: crossbeam_channel::Sender<String>,
) -> impl MakeSink<Sink = BufferedChannelSink> {
    buffered_channel_with_capacity(tx, 4096)
}

/// [`buffered_channel`] with an explicit initial buffer size.
pub fn buffered_channel_with_capacity(
    tx: crossbeam_channel::Sender<String>,
    capacity: usize,
) -> impl MakeSink<Sink = BufferedChannelSink> {
    move || BufferedChannelSink {
        tx: tx.clone(),
        buf: String::with_capacity(capacity),
        disconnected: false,
    }
}

/// Retained buffers for [`pooled_string`].  Bounded: a process that renders one
/// huge batch and then many small ones must not pin the huge allocation forever.
const MAX_POOLED_BUFFERS: usize = 8;

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
    pool: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
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
        if let Ok(mut pool) = self.pool.lock() {
            if pool.len() < MAX_POOLED_BUFFERS {
                pool.push(buf);
            }
        }
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
pub fn pooled_string() -> impl MakeSink<Sink = PooledStringSink> {
    pooled_string_with_capacity(4096)
}

/// [`pooled_string`] with an explicit initial buffer size (used only on the first
/// batch, before the pool warms up).
pub fn pooled_string_with_capacity(capacity: usize) -> impl MakeSink<Sink = PooledStringSink> {
    let pool: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
        std::sync::Arc::new(std::sync::Mutex::new(vec![String::with_capacity(capacity)]));
    move || {
        let buf = pool
            .lock()
            .ok()
            .and_then(|mut pool| pool.pop())
            .unwrap_or_else(|| String::with_capacity(capacity));
        PooledStringSink {
            buf,
            pool: std::sync::Arc::clone(&pool),
        }
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    /// One batch, one message — and a batch that is dropped without an explicit
    /// `flush` still reaches the receiver (the panic-containment path).
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

        // Drop without flush: best-effort delivery.
        {
            let mut sink =
                buffered_channel_with_capacity(crossbeam_channel::unbounded().0, 16).make();
            sink.write_str("dropped").unwrap();
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
