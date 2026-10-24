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

use std::fmt;
use std::io::{self, Write};

/// A destination for rendered diagnostic text.
///
/// Implementors decide what "writing" means: appending to a buffer, sending
/// over a channel, or calling a foreign callback.
///
/// A sink is created by [`MakeSink::make`] and dropped inside a single
/// `Emitter` batch method, so it never crosses a thread boundary and is
/// intentionally *not* required to be `Send`.  This is what lets
/// [`stderr()`] hold an [`io::StderrLock`] for the whole batch:
/// `io::StderrLock` is `!Send`, but it is only ever used on the thread that
/// created it.  Sinks that *are* `Send` (channels, strings) still work; the
/// emitter remains `Send` whenever its renderer and factory are.
pub trait DiagnosticSink {
    /// Write a string slice to the sink.
    fn write_str(&mut self, s: &str) -> io::Result<()>;

    /// Write formatted arguments to the sink.
    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> io::Result<()> {
        self.write_str(&args.to_string())
    }

    /// Flush any buffered output.  Defaults to a no-op.
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

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

impl StringSink {
    pub fn new() -> Self {
        StringSink { buf: String::new() }
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
        // Write straight into the internal buffer, avoiding a temporary `String`.
        let _ = self.buf.write_fmt(args);
        Ok(())
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

/// Test sink factory: produce a fresh [`StringSink`] per batch.
pub fn string() -> impl MakeSink<Sink = StringSink> {
    StringSink::new
}

/// LSP sink: forward rendered chunks over an mpsc channel.
pub struct ChannelSink(crossbeam_channel::Sender<String>);

impl DiagnosticSink for ChannelSink {
    fn write_str(&mut self, s: &str) -> io::Result<()> {
        self.0
            .send(s.to_owned())
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "channel closed"))
    }

    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> io::Result<()> {
        // Exactly one allocation: the formatted string sent over the channel.
        self.0
            .send(args.to_string())
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "channel closed"))
    }
}

/// LSP sink factory: clone the sender for each batch.
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
        // The blanket impl makes any `Write + Send` a sink.
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
}
