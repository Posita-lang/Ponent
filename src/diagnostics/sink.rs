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
//!   stay written.  For a **buffered** sink (`BufWriter`) that is only true if
//!   the emitter calls `flush()` before propagating a mid-batch error, and
//!   propagates `flush()` errors at batch end — `BufWriter`'s `Drop` silently
//!   *discards* unflushed bytes and cannot report the failure from `drop`.
//!   The emitter contract therefore carries the guarantee; this module only
//!   states it.
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
            // Ideally `io::const_error!` (message without the `Box`), but it is
            // still unstable on this toolchain (E0658), and `broken_pipe()`'s
            // payload-free form would drop the message a formatter bug needs.
            Err(_) => Err(io::Error::new(io::ErrorKind::Other, "formatter error")),
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
pub struct ChannelSink(crossbeam_channel::Sender<String>);

/// Payload-free `BrokenPipe`: recurring sends after the receiver dropped must
/// not allocate a fresh `Box<dyn Error>` each time.
#[inline]
fn broken_pipe() -> io::Error {
    io::Error::from(io::ErrorKind::BrokenPipe)
}

impl sealed::Sealed for ChannelSink {}

impl DiagnosticSink for ChannelSink {
    fn write_str(&mut self, s: &str) -> io::Result<()> {
        self.0.send(s.to_owned()).map_err(|_| broken_pipe())
    }

    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> io::Result<()> {
        // Exactly one allocation: the formatted string sent over the channel.
        self.0.send(args.to_string()).map_err(|_| broken_pipe())
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
    move || ChannelSink(tx.clone())
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

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
        let mut direct = Vec::new();
        let mut buffered = io::BufWriter::with_capacity(16, Vec::new());

        let chunks = ["hello ", "world\n", "line 2\n", "x"];
        for chunk in &chunks {
            direct.write_all(chunk.as_bytes()).unwrap();
            buffered.write_all(chunk.as_bytes()).unwrap();
        }
        // Disambiguate: `BufWriter` implements both `Write` and (via the
        // blanket impl) `DiagnosticSink`, each with its own `flush`.
        Write::flush(&mut buffered).unwrap();
        // The `DiagnosticSink` flush path (what the emitter calls) must forward
        // to `Write::flush`, not be a separate no-op.
        DiagnosticSink::flush(&mut buffered).unwrap();

        assert_eq!(direct, buffered.into_inner().unwrap());
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
        let mut channel_sink = ChannelSink(tx);
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
        let mut sink = ChannelSink(tx);
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
