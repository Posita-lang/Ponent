use std::borrow::Cow;
use std::io::{IoSlice, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use swrite::{SWrite, swrite};
use swallow_it::swallow;

#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};

use crate::diagnostics::error_code::ErrCode;
use crate::smallvec::SmallVec;
use compact_str::CompactString;

#[cfg(unix)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PendingAction {
    Close,
    Serve,
    TooLarge,
}

/// One-shot pending connection state .  Request bytes accumulate
/// non-blockingly in the poll loop; a connection never blocks the accept
/// thread while waiting for its request line or headers.
#[cfg(unix)]
struct PendingConn {
    stream: TcpStream,
    accepted_at: Instant,
    /// When the first byte arrived, if it has.  The 2 s partial-request window
    /// was anchored at `accepted_at`, so a client that connected, stalled under
    /// load, then dribbled a header had only the remainder of its budget while
    /// actively transmitting — and its slot churned.
    first_byte_at: Option<Instant>,
    buf: SmallVec<[u8; 512]>,
    scan: HeaderScan,
    /// Header terminator offset already found by `scan`; passed to
    /// `parse_request_head` so it does not re-search the whole buffer.
    head_end: Option<usize>,
}

#[cfg(unix)]
impl PendingConn {
    /// Silent preconnects expire after the first-byte budget; a partial
    /// request gets a longer but still bounded window.
    fn expiry(&self) -> Instant {
        match self.first_byte_at {
            // Silent preconnect: unchanged first-byte budget.
            None => self.accepted_at + FIRST_BYTE_READ_TIMEOUT,
            // Actively transmitting: the full transfer window from first byte.
            Some(t0) => t0 + Duration::from_secs(2),
        }
    }
}

/// Wrap `libc::epoll_ctl`, converting a non-zero return into `io::Error`.
#[cfg(target_os = "linux")]
fn epoll_ctl_checked(
    epfd: i32,
    op: i32,
    fd: i32,
    event: *mut libc::epoll_event,
) -> std::io::Result<()> {
    let rc = unsafe { libc::epoll_ctl(epfd, op, fd, event) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// Accept one connection already non-blocking.  Linux folds `accept` +
/// `set_nonblocking` + `CLOEXEC` into one `accept4` syscall; other Unix
/// targets keep the two-step fallback.
#[cfg(unix)]
fn accept_nonblocking(listener: &TcpListener) -> std::io::Result<TcpStream> {
    #[cfg(target_os = "linux")]
    unsafe {
        let fd = libc::accept4(
            listener.as_raw_fd(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
        );
        if fd < 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(TcpStream::from_raw_fd(fd))
        }
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        let (stream, _) = listener.accept()?;
        stream.set_nonblocking(true)?;
        Ok(stream)
    }
}

/// Restores a listener's original non-blocking mode on scope exit.
///
/// std exposes only `set_nonblocking(bool)`, no getter; `socket2::SockRef`
/// fills that gap on every platform.  A guard makes the restore unconditional:
/// normal return, `?`, and panic unwinding all restore the caller's mode.
struct RestoreNonblocking<'a> {
    listener: &'a TcpListener,
    /// The previous non-blocking bit, when the platform lets us read it.
    ///
    /// Unix queries it.  Winsock has **no** way to read the bit, so `None` means
    /// "unknown" and `Drop` leaves the socket alone.  The old code stored
    /// `false` there, i.e. it unconditionally restored *blocking* — turning a
    /// deliberately non-blocking listener into a blocking one on Windows, which
    /// is worse than not restoring at all (the caller that set it up owns the
    /// state on that platform).
    was_nonblocking: Option<bool>,
}

impl<'a> RestoreNonblocking<'a> {
    fn new(listener: &'a TcpListener, nonblocking: bool) -> std::io::Result<Self> {
        // Query first: if the query fails the listener is untouched.  `socket2`
        // is the dependency this file already carries for exactly this gap;
        // `.ok()` keeps today's documented `None` degradation on a platform that
        // cannot answer, so the cfg (and the mode leak it encoded on Windows) is
        // gone rather than duplicated.
        let was_nonblocking = socket2::SockRef::from(listener).nonblocking().ok();
        listener.set_nonblocking(nonblocking)?;
        Ok(Self {
            listener,
            was_nonblocking,
        })
    }
}

impl Drop for RestoreNonblocking<'_> {
    fn drop(&mut self) {
        // Destructors must not panic (especially during unwinding): a failed
        // restore is best-effort, not a secondary abort.
        if let Some(previous) = self.was_nonblocking {
            swallow!(self.listener.set_nonblocking(previous));
        }
    }
}

#[cfg(all(target_os = "linux", not(test)))]
fn enable_tcp_defer_accept(listener: &TcpListener) -> TcpDeferAcceptGuard {
    use std::os::fd::AsRawFd;
    let fd = listener.as_raw_fd();
    // Query first: if the read fails the socket is left alone and the guard
    // carries no restore token.
    let mut previous: libc::c_int = 0;
    let mut len = std::mem::size_of_val(&previous) as libc::socklen_t;
    // SAFETY: `fd` is a live TCP socket; `previous`/`len` describe a valid
    // `c_int` out-parameter for this option.
    let queried = unsafe {
        libc::getsockopt(
            fd,
            libc::IPPROTO_TCP,
            libc::TCP_DEFER_ACCEPT,
            std::ptr::from_mut(&mut previous).cast(),
            &mut len,
        )
    } == 0;
    // Only mutate the option when its previous value is known: the guard's whole
    // point is to restore the caller's socket, and a failed query used to still
    // set the option, leaving a permanent change with no restore token.
    if !queried {
        return TcpDeferAcceptGuard::no_restore(fd);
    }
    let one: libc::c_int = 1;
    // SAFETY: `listener` owns a live TCP socket fd; `one` is a valid
    // `c_int` and its size is exactly the optval length.  The call only
    // changes a socket option; a failure is a portability/backstop issue, and
    // the userspace first-byte budgets remain authoritative.
    let set_ok = unsafe {
        libc::setsockopt(
            fd,
            libc::IPPROTO_TCP,
            libc::TCP_DEFER_ACCEPT,
            std::ptr::from_ref(&one).cast(),
            std::mem::size_of_val(&one) as libc::socklen_t,
        )
    } == 0;
    TcpDeferAcceptGuard {
        fd,
        // A failed `setsockopt` changed nothing, so there is nothing to restore.
        previous: set_ok.then_some(previous),
    }
}

/// Restores a caller-owned listener's `TCP_DEFER_ACCEPT` when dropped.
///
/// The serving entry points document that they leave the listener as they found
/// it (see [`RestoreNonblocking`]), and `TCP_DEFER_ACCEPT` was the one socket
/// option mutated without a matching restore — a caller that reused the
/// listener inherited a silently changed accept policy.  Best-effort by design:
/// a failing `setsockopt` during unwinding must never panic, `previous` is
/// `None` when the option could not be read, and an fd the caller closed early
/// just makes the restore fail harmlessly.
#[cfg(all(target_os = "linux", not(test)))]
struct TcpDeferAcceptGuard {
    fd: std::os::fd::RawFd,
    previous: Option<libc::c_int>,
}

#[cfg(all(target_os = "linux", not(test)))]
impl TcpDeferAcceptGuard {
    /// A guard that restores nothing: the option was never changed (the query
    /// failed, or the `setsockopt` did).  Keeping the same type means the caller
    /// has one return path and no `Option<Guard>`.
    fn no_restore(fd: std::os::fd::RawFd) -> Self {
        TcpDeferAcceptGuard { fd, previous: None }
    }
}

#[cfg(all(target_os = "linux", not(test)))]
impl Drop for TcpDeferAcceptGuard {
    fn drop(&mut self) {
        let Some(previous) = self.previous else {
            return;
        };
        // SAFETY: the fd belongs to the caller's listener, which outlives the
        // call this guard was created in; the payload is the same `c_int` shape
        // the set call used.
        unsafe {
            libc::setsockopt(
                self.fd,
                libc::IPPROTO_TCP,
                libc::TCP_DEFER_ACCEPT,
                std::ptr::from_ref(&previous).cast(),
                std::mem::size_of_val(&previous) as libc::socklen_t,
            );
        }
    }
}

#[cfg(any(not(target_os = "linux"), test))]
fn enable_tcp_defer_accept(_listener: &TcpListener) {}

/// Serve the explanation for `code` on an already-bound `listener`.
/// The listener must have been bound to `127.0.0.1:<port>` by the caller
/// (e.g. via `TcpListener::bind("127.0.0.1:0")`).
///
/// Returns an error if no client sends an HTTP request within the accept
/// budget (10 s: a browser cold start, profile-lock waits, and extension
/// loading can take several seconds before the first request), so the
/// caller can surface a warning instead of silently pretending the page
/// was served.  Connections that never send a request — the browser's
/// speculative preconnects — are closed and the wait continues, so a
/// preconnect cannot consume the single page this call exists to serve.
/// The listener may be blocking or non-blocking on entry; the mode is restored
/// on every return path (queried via `socket2::SockRef` on Unix, with the
/// blocking fallback contract on non-Unix).
pub fn serve_explain(listener: &TcpListener, code: &str) -> std::io::Result<()> {
    // `ErrCode::new` carries a `debug_assert!` that aborts on unknown codes;
    // `code` here comes from the CLI (`ponent explain <CODE>`), so a typo
    // must not crash a debug build.  `explain_error_code` already reports
    // unknown codes gracefully — surface the same condition as an
    // `io::Error` so the caller can warn instead.
    //
    // Validate *before* switching the listener to non-blocking, so the early
    // return cannot leak the mode into the caller's listener.
    let ec = ErrCode::try_new(code)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;

    // Security: the explain page embeds error text derived from the compiler's
    // own tables, but it is still an HTTP endpoint — binding it to a
    // non-loopback interface would expose it to the whole network.  The sole
    // caller binds `127.0.0.1:0`; this check guards future callers.
    if !listener
        .local_addr()
        .map(|a| a.ip().is_loopback())
        .unwrap_or(false)
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "explain server: listener is not bound to a loopback address",
        ));
    }

    let _nonblocking_guard = RestoreNonblocking::new(listener, true)?;
    // Held in an `Option` so it can be restored at the moment it stops being useful —
    // see the `take()` in the accept loop.  A plain binding lived until function exit,
    // i.e. through the entire request drain, leaving every later pending connection under
    // the deferred wakeup the guard exists to release.
    let mut defer_accept = Some(enable_tcp_defer_accept(listener));

    let html = std::sync::Arc::<str>::from(build_explain_page(&ec));
    // only a request for this code's page consumes the one-shot server.
    // Use the *canonical* spelling: `try_new("e030")` accepted the code, so a
    // browser navigating to `/E030` is the same page — the raw CLI spelling
    // would 404 it and wait out the budget.
    let expected_path = std::sync::Arc::<str>::from(format!("/{}", ec.code()));
    let mut workers = JoinWorkers::default();

    // A browser cold start routinely takes several seconds before the first
    // request is sent; the old 2 s budget could expire while the browser was
    // still starting, leaving the page unserved.
    const ACCEPT_BUDGET: Duration = Duration::from_secs(10);
    // After the first page is delivered, keep the listener alive for See also
    // navigation instead of exiting immediately.  Tests use a short window so
    // the regression suite does not pay the production idle delay.
    const POST_SERVE_IDLE: Duration = if cfg!(test) {
        Duration::from_millis(500)
    } else {
        Duration::from_secs(10)
    };
    let deadline = Instant::now() + ACCEPT_BUDGET;
    let mut idle_deadline: Option<Instant> = None;
    let mut served = false;
    #[cfg(unix)]
    let delivered = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    #[cfg(not(unix))]
    let mut first_error: Option<std::io::Error> = None;
    // Keep the last `poll` failure so the give-up path can report the real
    // cause instead of a generic "no client connected" timeout.  `gave_up`
    // distinguishes that path from a normal deadline/idle exit, after which a
    // recovered transient error must not be reported as fatal.
    #[cfg(unix)]
    let mut last_poll_error: Option<std::io::Error> = None;
    #[cfg(unix)]
    let mut poll_gave_up = false;

    // Accept speculative preconnects into a poll-set instead of
    // servicing them serially.  A silent preconnect sends nothing, so `poll`
    // never marks it readable and it cannot delay the socket that carries the
    // real request; a socket that *has* bytes is served immediately.  The
    // pending set is bounded so a local process cannot grow it without bound.
    #[cfg(unix)]
    {
        const MAX_PENDING: usize = 32;
        let mut pending: Vec<PendingConn> = Vec::new();
        // Reuse one fd array across rounds: the loop runs up to 10×/s
        // and a fresh `Vec` per round is pure allocator churn.
        let mut fds: Vec<libc::pollfd> = Vec::with_capacity(1 + MAX_PENDING);
        let mut poll_errors = 0u32;
        // Repeated `poll` failures must back off instead of
        // exhausting the counter in microseconds.
        let mut poll_backoff_ms = 1u64;
        // consecutive `accept()` failures while the listener stays
        // readable (fd exhaustion) must back off instead of re-polling hot.
        let mut accept_errors = 0u32;
        while Instant::now() < deadline || idle_deadline.is_some() {
            let now = Instant::now();
            if !served && delivered.load(Ordering::Acquire) {
                served = true;
                idle_deadline = Some(now + POST_SERVE_IDLE);
            }
            if idle_deadline.is_some_and(|idle| now >= idle) {
                break;
            }
            if idle_deadline.is_none() && now >= deadline {
                break;
            }
            // [6]1: evict expired silent connections independently of socket
            // readiness.  `poll` returning 0 wakes the loop at the earliest
            // expiry, but an expired connection is not readable, so the
            // readiness-only sweep below would never recycle its slot.
            let now = Instant::now();
            let mut i = 0;
            while i < pending.len() {
                if now >= pending[i].expiry() {
                    let mut p = pending.swap_remove(i);
                    swallow!(p.stream.shutdown(Shutdown::Both));
                } else {
                    i += 1;
                }
            }
            fds.clear();
            // Only ask for the listener when a pending slot is
            // free.  At capacity new connections stay in the kernel backlog
            // instead of being accepted and immediately discarded.
            let listener_offset = usize::from(pending.len() < MAX_PENDING);
            if listener_offset == 1 {
                fds.push(libc::pollfd {
                    fd: listener.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                });
            }
            fds.extend(pending.iter().map(|p| libc::pollfd {
                fd: p.stream.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            }));
            // wake at the earliest pending expiry even with no readiness,
            // so a silent preconnect cannot pin the slot for the full budget.
            let earliest = pending.iter().map(PendingConn::expiry).min();
            let phase_deadline = idle_deadline.unwrap_or(deadline);
            let wait_until = earliest.map_or(phase_deadline, |e| e.min(phase_deadline));
            let remaining = wait_until.saturating_duration_since(Instant::now());
            // Poll for the whole remaining budget: the fd set is
            // level-triggered and nothing in the loop is time-driven except
            // the deadline, so a fixed 100 ms tick only burns wakeups.
            // Ceiling, not truncation: a positive sub-millisecond remainder
            // truncated to 0 turned the poll into a busy spin for the rest of
            // the interval.  `div_ceil` keeps the wait as short as the deadline
            // allows without ever polling with a zero timeout.
            let ms = usize::try_from(remaining.as_nanos().div_ceil(1_000_000))
                .unwrap_or(i32::MAX as usize)
                .clamp(1, i32::MAX as usize) as i32;
            // SAFETY: `fds` is a live, correctly sized array of `pollfd`; the
            // count is exactly its length.
            let rc = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, ms) };
            if rc < 0 {
                let e = std::io::Error::last_os_error();
                if e.kind() == std::io::ErrorKind::Interrupted {
                    // EINTR: `poll` was aborted by a signal; retry until the
                    // wall-clock deadline, exactly as `wait_readable` does.
                    continue;
                }
                // A transient `ENOMEM`/`EINTR`-class failure must not end
                // the wait while the browser may still connect.  Give up only
                // after a sustained, consecutive error run.
                poll_errors += 1;
                last_poll_error = Some(e);
                if poll_errors >= 100 {
                    poll_gave_up = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(poll_backoff_ms));
                poll_backoff_ms = (poll_backoff_ms * 2).min(50);
                continue;
            }
            poll_errors = 0;
            poll_backoff_ms = 1;
            if rc == 0 {
                continue;
            }
            // H2: `poll` sets POLLERR/POLLHUP/POLLNVAL regardless of the
            // requested events.  Without this a listener error is a 10 s hot
            // spin (POLLIN clear, immediate re-poll).
            if listener_offset == 1
                && fds[0].revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0
            {
                break;
            }
            if listener_offset == 1 && fds[0].revents & libc::POLLIN != 0 {
                // Accept every queued connection in one go; a burst of
                // preconnects would otherwise be accepted one deadline-tick
                // at a time.
                while pending.len() < MAX_PENDING {
                    match accept_nonblocking(listener) {
                        Ok(stream) => {
                            accept_errors = 0;
                            pending.push(PendingConn {
                                stream,
                                accepted_at: Instant::now(),
                                first_byte_at: None,
                                buf: SmallVec::new(),
                                scan: HeaderScan::new(),
                                head_end: None,
                            });
                        }
                        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                        Err(e) => {
                            // listener still POLLIN-readable but accept
                            // fails (EMFILE/ENFILE/...).  Sleeping here bounds
                            // the otherwise immediate re-poll to a hot spin.
                            accept_errors += 1;
                            // Cap the shift at 5 so the `.min(1000)` is reachable:
                            // with `min(4)` the maximum was 800 ms and the cap was
                            // dead code (the thread path's ceiling is 1 s).
                            let ms = (50u64 << accept_errors.min(5)).min(1000);
                            std::thread::sleep(Duration::from_millis(ms));
                            break;
                        }
                    }
                }
            }
            // advance every ready pending connection without blocking.  Request
            // bytes are accumulated here; a connection is handed to a worker
            // only once its header block is complete.  Silent preconnects
            // expire on their per-connection deadline instead of pinning a
            // slot.
            let polled = fds.len().saturating_sub(listener_offset);
            let mut remove: SmallVec<[(usize, PendingAction); MAX_PENDING]> = SmallVec::new();
            for i in 0..polled.min(pending.len()) {
                let revents = fds[listener_offset + i].revents;
                if revents == 0 {
                    continue;
                }
                if revents & libc::POLLNVAL != 0 {
                    remove.push((i, PendingAction::Close));
                    continue;
                }
                let mut action = None;
                loop {
                    let mut tmp = [0u8; 4096];
                    match pending[i].stream.read(&mut tmp) {
                        Ok(0) => {
                            action = Some(PendingAction::Close);
                            break;
                        }
                        Ok(n) => {
                            // First byte: start the transfer window (see `expiry`).
                            swallow!(pending[i].first_byte_at.get_or_insert_with(Instant::now));
                            pending[i].buf.extend_from_slice(&tmp[..n]);
                            let progress = pending[i].scan.scan(&pending[i].buf);
                            match progress {
                                HeaderProgress::Complete(end) => {
                                    pending[i].head_end = Some(end);
                                    action = Some(PendingAction::Serve);
                                    break;
                                }
                                HeaderProgress::TooLargeHeader | HeaderProgress::TooLargeLine => {
                                    action = Some(PendingAction::TooLarge);
                                    break;
                                }
                                HeaderProgress::Incomplete => {}
                            }
                            if n < tmp.len() {
                                break;
                            }
                        }
                        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                        // Every other read/write loop in this file retries on
                        // EINTR, and this is the one that runs on a foreground
                        // terminal: `SIGWINCH` (a window resize), `SIGCHLD` or
                        // `SIGCONT` landing mid-read used to kill a connection
                        // that had already begun sending its request.
                        Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(_) => {
                            action = Some(PendingAction::Close);
                            break;
                        }
                    }
                }
                if let Some(action) = action {
                    remove.push((i, action));
                }
            }

            let mut handoff: SmallVec<
                [(
                    TcpStream,
                    SmallVec<[u8; 512]>,
                    PendingAction,
                    Option<RequestHead>,
                ); 8],
            > = SmallVec::new();
            for &(i, action) in remove.iter().rev() {
                let p = pending.swap_remove(i);
                if action == PendingAction::Close {
                    let mut stream = p.stream;
                    swallow!(stream.shutdown(Shutdown::Both));
                } else {
                    let head = if action == PendingAction::Serve {
                        parse_request_head(&p.buf, p.head_end)
                    } else {
                        None
                    };
                    handoff.push((p.stream, p.buf, action, head));
                }
            }

            if !handoff.is_empty() {
                for (mut stream, buf, action, head) in handoff.drain(..) {
                    if action == PendingAction::TooLarge {
                        if !workers.has_capacity() {
                            // Overload shed: answering here would hold the
                            // reactor for the client's write timeout, which is
                            // the stall the worker pool exists to prevent, so
                            // close instead.
                            swallow!(stream.shutdown(Shutdown::Both));
                            continue;
                        }
                        let handle =
                            std::thread::Builder::new()
                                .stack_size(128 * 1024)
                                .spawn(move || {
                                    if stream.set_nonblocking(false).is_err() {
                                        swallow!(stream.shutdown(Shutdown::Both));
                                        return;
                                    }
                                    swallow!(stream.set_write_timeout(Some(Duration::from_secs(1))));
                                    swallow!(write_header_too_large(&mut stream));
                                    linger_after_half_close(&mut stream);
                                });
                        if let Ok(handle) = handle {
                            workers.push(handle);
                        }
                        continue;
                    }
                    if !workers.has_capacity() {
                        // Eight requests already in flight: shed this one
                        // without blocking the accept loop (see has_capacity).
                        swallow!(stream.shutdown(Shutdown::Both));
                        continue;
                    }
                    let html = std::sync::Arc::clone(&html);
                    let expected = std::sync::Arc::clone(&expected_path);
                    let delivered = std::sync::Arc::clone(&delivered);
                    let handle =
                        std::thread::Builder::new()
                            .stack_size(128 * 1024)
                            .spawn(move || {
                                let r =
                                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                        serve_page_after_prefix(
                                            &mut stream,
                                            html.as_ref(),
                                            Some(expected.as_ref()),
                                            &buf,
                                            head,
                                        )
                                    }));
                                if matches!(r, Ok(Ok(true))) {
                                    delivered.store(true, Ordering::Release);
                                }
                            });
                    if let Ok(handle) = handle {
                        workers.push(handle);
                    }
                }
            }
        }
        for p in pending.drain(..) {
            let mut stream = p.stream;
            swallow!(stream.set_nonblocking(true));
            swallow!(drain_request(&mut stream, &[], Duration::from_millis(20)));
            swallow!(stream.shutdown(Shutdown::Both));
        }
    }

    // Non-unix targets have no raw `poll`; keep the serial accept loop.  The
    // deadline still bounds a silent preconnect.
    #[cfg(not(unix))]
    loop {
        let now = Instant::now();
        if now >= idle_deadline.unwrap_or(deadline) {
            break;
        }
        match listener.accept() {
            Ok((mut stream, _)) => {
                // The connection the deferral was waiting for has arrived: restore the
                // listener's previous setting *now*, before draining, so the next
                // connection is accepted eagerly.
                drop(defer_accept.take());
                match serve_page_expected(&mut stream, html.as_ref(), Some(expected_path.as_ref()))
                {
                    Ok(true) => {
                        served = true;
                        idle_deadline = Some(now + POST_SERVE_IDLE);
                    }
                    Ok(false) => continue,
                    Err(e) => {
                        if first_error.is_none() {
                            first_error = Some(e);
                        }
                        continue;
                    }
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => {
                if first_error.is_none() {
                    first_error = Some(e);
                }
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
        }
    }
    // Browsers often open several speculative preconnect connections
    // alongside the one they actually use; leaving them unaccepted gets them
    // RST on process exit, which the browser can surface as a connection
    // error.  The listener is still non-blocking here, so drain *every*
    // pending connection (loop until `WouldBlock`) — each drain is best-effort
    // and short-budget, so exit is not delayed by more than the sockets
    // actually idle.
    //
    // Bound the drain: a hostile local process that fills the listen backlog
    // would otherwise pin exit for ~50 ms per pending socket.  Browsers open
    // 2–4 preconnects, comfortably under this cap.  `drained` lives outside
    // the loop — declaring it inside reset the counter every iteration and
    // made the cap ineffective.
    const DRAIN_MAX: usize = 16;
    let mut drained = 0usize;
    loop {
        match listener.accept() {
            Ok((mut extra, _)) => {
                // The accepted socket may be blocking; a preconnect that sends
                // nothing would otherwise hold `drain_request` on a blocking
                // `read` for the full 1 s budget (or indefinitely if the peer
                // never closes).
                //
                // Force *non-blocking* so the drain consumes only bytes that
                // have already arrived, so a silent peer cannot hold it on a
                // blocking `read`.  The mode change is set, not merely
                // propagated: if it fails, `read` would stay blocking with no
                // timeout and the drain could wait forever, so skip the drain
                // and just close.
                if extra.set_nonblocking(true).is_err() {
                    swallow!(extra.shutdown(Shutdown::Both));
                    drained += 1;
                    if drained >= DRAIN_MAX {
                        break;
                    }
                    continue;
                }
                swallow!(drain_request(&mut extra, &[], Duration::from_millis(50)));
                swallow!(extra.shutdown(Shutdown::Both));
                drained += 1;
                if drained >= DRAIN_MAX {
                    break;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(_) => break, // drain is best-effort; never fail the page result
        }
    }

    workers.join_all();
    #[cfg(unix)]
    if !served && delivered.load(Ordering::Acquire) {
        served = true;
    }
    let result = if served {
        Ok(())
    } else {
        #[cfg(not(unix))]
        if let Some(error) = first_error {
            return Err(error);
        }
        #[cfg(unix)]
        if poll_gave_up {
            // The wait was cut short by a sustained `poll` failure, not by
            // the deadline: report the underlying error instead of pretending
            // no client connected.
            let detail = match last_poll_error {
                Some(e) => {
                    format!("explain server: gave up after 100 consecutive poll failures: {e}")
                }
                None => "explain server: gave up after 100 consecutive poll failures".to_owned(),
            };
            return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, detail));
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "explain server: no client connected within ~10s",
        ))
    };
    // `_nonblocking_guard` restores the caller's original mode on drop.
    result
}

/// Parse a `Content-Length` header out of a raw header block.
///
/// Returns `0` when the header is absent or malformed (the caller is a GET-only
/// server, so the conservative default is "no body").  The value is capped at
/// [`MAX_BODY_DRAIN`] so a hostile client cannot make the bounded drain read for
/// longer than its wall-clock budget trying to satisfy an absurd length.
///
/// Deliberately minimal: a duplicate `Content-Length` would let the last value
/// win, and `Transfer-Encoding: chunked` is not honoured (a chunked body is
/// simply not drained, so the connection is closed instead of kept alive). Both
/// are accepted risks for a loopback diagnostic server that serves browsers,
/// not a general-purpose origin server .
/// Resource cap for a declared request body.
const MAX_BODY_DRAIN: usize = 8 * 1024 * 1024;

/// Framing facts extracted from the header block in ONE pass.
///
/// The old two-function version (`parse_content_length` + a separate
/// `Transfer-Encoding` scan) could also turn a conflicting/malformed or
/// oversized `Content-Length` into a zero-length body, which `drain_request`
/// then reported as `Complete` — a false completion that let the RST/RST-free
/// close happen with request bytes still unread .
/// Resource caps shared by every HTTP backend.
/// Idle window of a reused keep-alive connection, shared by every serve loop.
///
/// Defined twice before (once per loop), so a change to one left the other silently
/// different: the persistent reactor and the one-shot/poll paths would then release idle
/// sockets on different schedules — invisible until a browser reuses a connection the
/// other mode would already have closed.
const KEEP_ALIVE_IDLE: Duration = Duration::from_secs(2);

const MAX_HEADER: usize = 64 * 1024;
/// Linger after a one-shot response is already on the wire: long enough for a
/// local browser to drain the final bytes (avoiding an RST that discards the
/// response), short enough to bound the one-shot exit latency.  Also used by
/// the 431/malformed half-close paths, which otherwise closed after only
/// 50 ms and could truncate a large error body.
const ONE_SHOT_LINGER: Duration = Duration::from_millis(150);

#[derive(Default)]
struct JoinWorkers(Vec<std::thread::JoinHandle<()>>);

impl JoinWorkers {
    const MAX_WORKERS: usize = 8;

    /// Reap finished workers and report whether one more may be spawned.
    ///
    /// The capacity check has to happen *before* `spawn`, not inside `push`: a
    /// running worker may be draining a request body, writing to a client that
    /// is not reading, or inside the 150 ms linger, so joining it from the
    /// reactor thread reintroduces exactly the head-of-line blocking the
    /// polling loop exists to avoid.  Overload is therefore shed by the caller.
    ///
    /// Detaching a finished handle is safe — the thread has already exited.
    fn has_capacity(&mut self) -> bool {
        self.0.retain(|h| !h.is_finished());
        self.0.len() < Self::MAX_WORKERS
    }

    /// Adopt a freshly spawned worker.  The caller must have taken a slot with
    /// [`Self::has_capacity`] first, so this never has to join anything.
    fn push(&mut self, handle: std::thread::JoinHandle<()>) {
        debug_assert!(
            self.0.len() < Self::MAX_WORKERS,
            "push without a has_capacity slot"
        );
        self.0.push(handle);
    }

    fn join_all(&mut self) {
        for handle in self.0.drain(..) {
            swallow!(handle.join());
        }
    }
}

impl Drop for JoinWorkers {
    fn drop(&mut self) {
        self.join_all();
    }
}
const MAX_REQUEST_LINE: usize = 8 * 1024;
/// Bound one reactor wakeup's accept work so a sustained connect flood cannot
/// starve established reads/writes/deadlines; level-triggered epoll resumes.
const ACCEPT_BATCH: usize = 64;
/// Consecutive non-transient `accept` failures before the persistent server
/// reports the listener as dead instead of spinning forever.
const MAX_CONSECUTIVE_ACCEPT_ERRORS: usize = 100;

/// Incremental header scanner shared by PendingConn, `drain_request_tracked`
/// and the epoll reactor.  Only the newly appended suffix (plus the 3-byte
/// carry) is searched for a terminator, so a one-byte-at-a-time client is
/// linear instead of quadratic.  The same type also enforces the request-line
/// and header caps so the backends cannot drift.
#[derive(Debug)]
struct HeaderScan {
    scanned: std::cell::Cell<usize>,
    line_scanned: std::cell::Cell<usize>,
    line_nl: std::cell::Cell<Option<usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeaderProgress {
    Incomplete,
    Complete(usize),
    TooLargeHeader,
    TooLargeLine,
}

impl HeaderScan {
    const fn new() -> Self {
        HeaderScan {
            scanned: std::cell::Cell::new(0),
            line_scanned: std::cell::Cell::new(0),
            line_nl: std::cell::Cell::new(None),
        }
    }

    fn reset(&self) {
        self.scanned.set(0);
        self.line_scanned.set(0);
        self.line_nl.set(None);
    }

    /// Scan a buffer that begins at the request line.
    fn scan(&self, buf: &[u8]) -> HeaderProgress {
        match self.line_nl.get() {
            Some(position) if position >= MAX_REQUEST_LINE => {
                return HeaderProgress::TooLargeLine;
            }
            Some(_) => {}
            None => {
                let from = self.line_scanned.get().min(buf.len());
                match memchr::memchr(b'\n', &buf[from..]) {
                    Some(offset) => {
                        let position = from + offset;
                        self.line_nl.set(Some(position));
                        self.line_scanned.set(buf.len());
                        if position >= MAX_REQUEST_LINE {
                            return HeaderProgress::TooLargeLine;
                        }
                    }
                    None => {
                        self.line_scanned.set(buf.len());
                        if buf.len() > MAX_REQUEST_LINE {
                            return HeaderProgress::TooLargeLine;
                        }
                    }
                }
            }
        }
        self.scan_terminator(buf)
    }

    /// Scan a buffer that already had its request line consumed (the thread
    /// drain / one-shot prefix).  A lone blank line is a valid empty header
    /// block and must not be rejected by the "needs two terminators" rule.
    fn scan_after_request_line(&self, buf: &[u8]) -> HeaderProgress {
        if buf.starts_with(b"\r\n") {
            return HeaderProgress::Complete(2);
        }
        if buf.starts_with(b"\n") {
            return HeaderProgress::Complete(1);
        }
        self.scan_terminator(buf)
    }

    fn scan_terminator(&self, buf: &[u8]) -> HeaderProgress {
        let from = self.scanned.get().saturating_sub(3).min(buf.len());
        let end = header_end(&buf[from..]).map(|end| from + end);
        self.scanned.set(buf.len());
        match end {
            Some(end) if end <= MAX_HEADER => HeaderProgress::Complete(end),
            Some(_) => HeaderProgress::TooLargeHeader,
            None if buf.len() > MAX_HEADER => HeaderProgress::TooLargeHeader,
            None => HeaderProgress::Incomplete,
        }
    }
}

/// End offset (just past the blank line) of a complete header block,
/// accepting both CRLFCRLF and bare LFLF.  The thread path is deliberately
/// LF-lenient; the reactor must not reject a request the thread path serves.
fn header_end(buf: &[u8]) -> Option<usize> {
    // One forward scan, not two `memmem` passes.  Both old finders started at offset 0, so
    // a head terminated by `\r\n\r\n` scanned everything before it twice and then took a
    // minimum; a single `memchr` cursor with a short look-back finds the *smallest* end
    // directly, which is what the minimum computed.
    //
    // Truth table, for a `\n` at index `i` (the head ends at `i + 1`):
    //   `...\r\n\r\n`  → `buf[i-3..i] == "\r\n\r"`            CRLFCRLF
    //   `......\n\n`    → `buf[i-1] == b'\n'`                  LFLF
    //   `...\r\n\n`    → also `buf[i-1] == b'\n'`             CRLF then a bare LF
    //                                                     terminator (the old
    //                                                     `\n\n` finder matched this)
    // A single trailing `\n` (`...\r\n` at end of buffer) terminates nothing: a head
    // needs the blank line, so one `\n` after the last field is not enough.
    let mut from = 0usize;
    while let Some(rel) = memchr::memchr(b'\n', &buf[from..]) {
        let i = from + rel;
        if i >= 3 && &buf[i - 3..i] == b"\r\n\r" {
            return Some(i + 1);
        }
        if i >= 1 && buf[i - 1] == b'\n' {
            return Some(i + 1);
        }
        from = i + 1;
    }
    None
}

#[derive(Debug, Clone, Copy)]
struct HeaderFraming {
    content_length: Option<usize>,
    has_transfer_encoding: bool,
    /// Conflicting duplicates, a malformed value, or a body over the cap: the
    /// framing cannot be trusted, so `Complete` may only come from peer EOF.
    ambiguous: bool,
    /// A header line with no `:` (excluding the request line and the blank
    /// terminator): the head is syntactically broken, so a `Content-Length`-like
    /// line cannot be distinguished from junk.  Kept separate from
    /// `has_transfer_encoding` — "unsupported but well-formed" and "not even a
    /// header" deserve different diagnosis — but both make the connection
    /// non-reusable, because a silently skipped framing field would otherwise
    /// turn a malformed request into an unambiguous empty-body one.
    malformed: bool,
    oversized: bool,
    /// H3: `Connection: close` appears anywhere in the complete header block
    /// (case-insensitive, comma-separated token list, multiple occurrences).
    connection_close: bool,
}

/// Body framing decision, derived once from the header scan.
///
/// The three backends used to answer "is the body unframed?" with
/// `has_transfer_encoding || ambiguous || oversized` and "is there no body?"
/// with `content_length.map_or(true, |n| n == 0)` independently — exactly the
/// shape where they can drift apart.  Both questions now have one answer, and
/// the boolean combinations that implied a fourth state are enumerated instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestBody {
    /// No body bytes remain to be read.
    Empty,
    /// A declared length with `remaining` bytes not yet read.
    Fixed { remaining: usize },
    /// Chunked/unsupported transfer coding, conflicting or malformed
    /// `Content-Length`, or a body over the cap.  Only peer EOF (or the drain
    /// budget) can complete it, so it must never be treated as reusable.
    Unframed,
}

impl HeaderFraming {
    /// Classify the body given how many body bytes already arrived in the same
    /// read as the head.
    fn body(self, already_read: usize) -> RequestBody {
        if self.has_transfer_encoding || self.ambiguous || self.malformed || self.oversized {
            return RequestBody::Unframed;
        }
        match self
            .content_length
            .unwrap_or(0)
            .saturating_sub(already_read)
        {
            0 => RequestBody::Empty,
            remaining => RequestBody::Fixed { remaining },
        }
    }

    /// Derive the framing/keep-alive decision **once**.
    ///
    /// The backends used to recompute "is there a body to drain?" and "may this
    /// connection be reused?" independently — `drain_request_tracked`, the epoll
    /// dispatcher and the request-head handoff each spelled the predicate out —
    /// which is exactly the shape where the two answers drift apart (a
    /// connection reused with an unread body gets RST; one closed unnecessarily
    /// loses keep-alive).  Everything framing-derived now comes from here; a
    /// caller may only add its *own* vetoes (the responder's status, whether
    /// pipelined bytes follow).
    ///
    /// `already_read` counts body bytes that arrived in the same read as the
    /// head; `http11` is the request's protocol version.
    fn decide(self, already_read: usize, http11: bool) -> RequestDecision {
        let body = self.body(already_read);
        let body_remaining = match body {
            RequestBody::Empty => Some(0),
            RequestBody::Fixed { remaining } => Some(remaining),
            // Framing cannot be trusted: only peer EOF (or the drain budget)
            // ends the body, so there is no length to report.
            RequestBody::Unframed => None,
        };
        let must_close = self.connection_close || !matches!(body, RequestBody::Empty);
        RequestDecision {
            body_remaining,
            keep_alive_allowed: http11 && !must_close,
            must_close,
        }
    }
}

/// Framing-derived connection decision — see [`HeaderFraming::decide`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RequestDecision {
    /// Body bytes still to read; `None` when the framing is untrusted
    /// ([`RequestBody::Unframed`]).
    body_remaining: Option<usize>,
    /// Framing alone permits another request on this connection (HTTP/1.1, no
    /// `Connection: close`, empty body).  Callers AND their own vetoes on top.
    keep_alive_allowed: bool,
    /// The connection must be closed after the response, whatever the responder
    /// decides.
    must_close: bool,
}

/// One parsed request head shared between the poll handoff and the worker.
#[derive(Debug, Clone)]
struct RequestHead {
    line: CompactString,
    head_end: usize,
    framing: HeaderFraming,
}

fn parse_request_head(buf: &[u8], head_end: Option<usize>) -> Option<RequestHead> {
    let nl = memchr::memchr(b'\n', buf)?;
    let raw = &buf[..nl];
    let raw = raw.strip_suffix(b"\r").unwrap_or(raw);
    let line = simdutf8::basic::from_utf8(raw).ok()?;
    let head_end = head_end.or_else(|| header_end(buf))?;
    if head_end > MAX_HEADER {
        return None;
    }
    Some(RequestHead {
        line: line.into(),
        head_end,
        framing: scan_framing(&buf[..head_end], true),
    })
}

/// Parse `Content-Length`/`Transfer-Encoding`/`Connection` from a complete
/// header block.
///
/// Identical duplicate `Content-Length` values are accepted; conflicting
/// duplicates set `ambiguous`, and either that or any `Transfer-Encoding`
/// forces close + drain instead of keep-alive.  This is a deliberate
/// smuggling-adjacent leniency: it never leaves a reusable connection with
/// unread body bytes.
fn scan_framing(headers: &[u8], first_line_is_request: bool) -> HeaderFraming {
    let mut framing = HeaderFraming {
        content_length: None,
        has_transfer_encoding: false,
        ambiguous: false,
        malformed: false,
        oversized: false,
        connection_close: false,
    };
    for (index, line) in headers.split(|&b| b == b'\n').enumerate() {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        // Trust the caller.  This used to guess by shape ("has a space, has no
        // colon"), which silently skipped a *malformed first header* such as
        // `Content-Length 5`: no `Content-Length` was parsed, the body was framed
        // as empty, and an HTTP/1.1 connection could be reused with the client's
        // body still inbound — a request-smuggling shape.  `first_line_is_request`
        // is true exactly when the caller's slice still starts with the request
        // line (`buf[..head_end]`), false for a headers-only block.
        if index == 0 && first_line_is_request {
            continue;
        }
        let Some(colon) = line.iter().position(|&b| b == b':') else {
            // A header line without a colon is not "a header we ignore": if it
            // was meant as `Content-Length 100`, skipping it silently would
            // frame the request as an unambiguous empty body.  Record it so the
            // connection is drained and closed instead of reused.
            framing.malformed = true;
            continue;
        };
        let name_raw = &line[..colon];
        if name_raw.first().is_some_and(|&b| b == b' ' || b == b'\t') {
            // RFC 7230 §3.2.4 forbids leading whitespace (obsolete line fold).
            framing.ambiguous = true;
            continue;
        }
        if name_raw.last().is_some_and(|&b| b == b' ' || b == b'\t') {
            // RFC 7230 §3.2.4: whitespace before the colon is forbidden.
            framing.ambiguous = true;
            continue;
        }
        if name_raw.iter().any(|&b| b == b' ' || b == b'\t') {
            // `Content-Length x: 5` — interior whitespace means this is not a
            // field name at all.  Ignoring it would frame the body as empty, so
            // mark the framing untrustworthy (drain and close).
            framing.malformed = true;
            continue;
        }
        let name = name_raw.trim_ascii();
        let value = line[colon + 1..].trim_ascii();
        if name.eq_ignore_ascii_case(b"content-length") {
            if value.is_empty() || !value.iter().all(u8::is_ascii_digit) {
                // `"+5"`, an empty value, or a non-digit: framing is broken.
                framing.ambiguous = true;
                continue;
            }
            let n = value.iter().fold(0usize, |n, &b| {
                n.saturating_mul(10).saturating_add((b - b'0') as usize)
            });
            if let Some(prev) = framing.content_length {
                if prev != n {
                    framing.ambiguous = true;
                }
            } else {
                framing.content_length = Some(n);
            }
        } else if name.eq_ignore_ascii_case(b"transfer-encoding") {
            // `identity` is RFC 9112 §6.1's "no transfer coding" token: it leaves the
            // framing provable, yet any TE header used to set this flag, so
            // `DrainOutcome` fell back to "wait for peer EOF" — and a keep-alive client
            // never sends EOF, so every such request sat out the full 1 s read budget.
            // A list that names any *real* coding (`chunked`, or `identity, chunked`) is
            // still unprovable and must keep the flag.
            let only_identity = value
                .split(|&b| b == b',')
                .map(|token| token.trim_ascii())
                .filter(|token| !token.is_empty())
                .all(|token| token.eq_ignore_ascii_case(b"identity"));
            if !only_identity {
                framing.has_transfer_encoding = true;
            }
        } else if name.eq_ignore_ascii_case(b"connection") {
            // Token list: `keep-alive, close`, extra whitespace, mixed case.
            if value
                .split(|&b| b == b',')
                .any(|token| token.trim_ascii().eq_ignore_ascii_case(b"close"))
            {
                framing.connection_close = true;
            }
        }
    }
    if framing.content_length.is_some_and(|n| n > MAX_BODY_DRAIN) {
        framing.oversized = true;
    }
    framing
}

/// How completely a request was drained from the socket.
///
/// A plain `bool` conflated "a header block arrived" with "the whole request
/// was consumed": the body drain can be cut short by the wall-clock budget,
/// and closing with unread inbound bytes makes the kernel send RST — the exact
/// failure the drain exists to avoid .
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DrainOutcome {
    /// No complete header block was observed.
    NoRequest,
    /// The header block exceeded `MAX_HEADER` (or the request line exceeded
    /// its cap): answer 431 instead of silently closing.
    TooLarge,
    /// The header block arrived but the announced body was not fully drained.
    HeadersOnly,
    /// Header block and any `Content-Length` body (and pipelined residue) were
    /// fully consumed; a close now sends FIN, not RST.
    Complete,
}

/// Clamp the socket read timeout to the remaining wall-clock budget, so a
/// blocking `read` cannot outlive the declared drain deadline.
fn arm_read_timeout_until(stream: &mut TcpStream, deadline: Instant) -> std::io::Result<()> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    stream.set_read_timeout(Some(remaining.max(Duration::from_millis(1))))
}

/// Read and discard an HTTP request through its `\r\n\r\n` terminator and any
/// body announced by `Content-Length`, until the peer closes or the
/// wall-clock budget expires.  Shared by [`serve_page`], the persistent
/// handler, and the speculative-connection drain in [`serve_explain`].
///
/// Only HTTP header framing is recognised.  The request line and the blank
/// line after the headers are LF-lenient (`\n\n` and a lone `\n` are accepted,
/// matching the tests and the incremental scanner); CRLF remains the canonical
/// browser form.
///
/// Accumulating the chunks makes a terminator split across *any* number of
/// reads correct.  The scan itself is incremental : only bytes appended by
/// the latest read are searched, with a 3-byte carry, so a dribbling client
/// costs O(n) total scan work instead of O(n²).
/// Drain a request whose framing cannot be established (chunked, ambiguous,
/// oversized) until peer EOF or the deadline.  One canonical loop for
/// `drain_after_head` and `drain_request_tracked`.
fn drain_unframed(stream: &mut TcpStream, deadline: Instant) -> DrainOutcome {
    let mut sink = [0u8; 4096];
    loop {
        if Instant::now() >= deadline {
            return DrainOutcome::HeadersOnly;
        }
        if arm_read_timeout_until(stream, deadline).is_err() {
            return DrainOutcome::HeadersOnly;
        }
        match stream.read(&mut sink) {
            Ok(0) => return DrainOutcome::Complete,
            Ok(_) => {}
            Err(ref e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                return DrainOutcome::HeadersOnly;
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return DrainOutcome::HeadersOnly,
        }
    }
}

fn drain_after_head(
    stream: &mut TcpStream,
    framing: HeaderFraming,
    already_read: usize,
    budget: Duration,
) -> DrainOutcome {
    let deadline = Instant::now() + budget;
    if matches!(framing.body(already_read), RequestBody::Unframed) {
        // One home for unframed-drain semantics.  This was a byte-identical
        // inline copy of `drain_unframed`, so a timeout-budget or `EINTR` fix
        // applied to one copy silently diverged the other — the exact drift
        // shape the framing decision and `HeaderScan` were consolidated to
        // avoid.
        return drain_unframed(stream, deadline);
    }
    let mut sink = [0u8; 4096];
    let mut remaining = framing
        .content_length
        .unwrap_or(0)
        .saturating_sub(already_read);
    while remaining > 0 {
        if Instant::now() >= deadline {
            return DrainOutcome::HeadersOnly;
        }
        if arm_read_timeout_until(stream, deadline).is_err() {
            return DrainOutcome::HeadersOnly;
        }
        let want = remaining.min(sink.len());
        match stream.read(&mut sink[..want]) {
            Ok(0) => return DrainOutcome::HeadersOnly,
            Ok(n) => remaining = remaining.saturating_sub(n),
            Err(ref e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                return DrainOutcome::HeadersOnly;
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return DrainOutcome::HeadersOnly,
        }
    }
    DrainOutcome::Complete
}

fn drain_or_pre(
    stream: &mut TcpStream,
    prefix: &[u8],
    pre: Option<DrainOutcome>,
    budget: Duration,
) -> DrainOutcome {
    pre.unwrap_or_else(|| drain_after_request_line(stream, prefix, budget))
}

/// Drain a request whose bytes *after* the request line are in `prefix`.
///
/// `drain_request_tracked`'s own doc records the bug that motivated the
/// `prefix_has_request_line` parameter: keeping it pinned to `true` made
/// `scan_framing` skip the first *header* line as a request line, and every
/// production caller of this shape passes post-line bytes (the routed one-shot
/// path and `read_request_line`'s drains).  Encoding the meaning in the wrapper
/// is what stops the next caller from inheriting the flag by habit.
fn drain_after_request_line(
    stream: &mut TcpStream,
    prefix: &[u8],
    budget: Duration,
) -> DrainOutcome {
    drain_with(
        stream,
        prefix,
        budget,
        DrainOptions {
            prefix_has_request_line: false,
            report_head: false,
        },
    )
    .0
}

/// The two flags that actually differ between the drain wrappers.
///
/// Every wrapper used to repeat the same three `&mut` scratch bindings and the same
/// nine-argument call in its own body, so four copies of "drain and discard the trackers"
/// existed; only these two things ever varied.
#[derive(Clone, Copy)]
struct DrainOptions {
    /// The prefix still begins with the request line (the test-only whole-head path).
    prefix_has_request_line: bool,
    /// The caller needs to know whether the request line named `HEAD`.
    report_head: bool,
}

/// The single body behind the drain wrappers.
fn drain_with(
    stream: &mut TcpStream,
    prefix: &[u8],
    budget: Duration,
    opts: DrainOptions,
) -> (DrainOutcome, bool) {
    let mut tail_len = 0;
    let mut wants_close = false;
    let mut is_head = false;
    let outcome = drain_request_tracked(
        stream,
        prefix,
        budget,
        &mut tail_len,
        &mut wants_close,
        &mut is_head,
        opts.prefix_has_request_line,
        None,
        0,
    );
    (outcome, if opts.report_head { is_head } else { false })
}

/// Drain a request whose prefix still **begins with the request line** (the
/// test-only whole-head path).
///
/// The flag matters: [`drain_after_request_line`]'s `false` is right for every
/// production caller, but this one passes the unparsed buffer, and treating its
/// request line as a colonless header marked the framing untrusted — which
/// drained unframed (read to EOF/1 s) where the request already had no body.
fn drain_whole_head(stream: &mut TcpStream, prefix: &[u8], budget: Duration) -> DrainOutcome {
    // Same flags as `drain_request`; only the *caller's* prefix contract differs (this
    // one is the test-only whole-head path).  The names stay because they document the
    // prefix shape at each call site.
    drain_with(
        stream,
        prefix,
        budget,
        DrainOptions {
            prefix_has_request_line: true,
            report_head: false,
        },
    )
    .0
}

/// Drain a request whose bytes arrive fresh on `stream` (empty-prefix paths).
fn drain_request(stream: &mut TcpStream, prefix: &[u8], budget: Duration) -> DrainOutcome {
    drain_with(
        stream,
        prefix,
        budget,
        DrainOptions {
            prefix_has_request_line: true,
            report_head: false,
        },
    )
    .0
}

/// [`drain_request`] plus whether the request line named `HEAD`.
fn drain_request_with_head(
    stream: &mut TcpStream,
    prefix: &[u8],
    budget: Duration,
) -> (DrainOutcome, bool) {
    drain_with(
        stream,
        prefix,
        budget,
        DrainOptions {
            prefix_has_request_line: true,
            report_head: true,
        },
    )
}

/// [`drain_request`], additionally reporting how many bytes past this
/// request's body were already read (a pipelined next request).  The
/// keep-alive loop forces a close when it is non-zero instead of silently
/// discarding the second request .
///
/// `is_head` is derived from `collected.starts_with(b"HEAD ")`, and
/// `collected` is seeded with `prefix`.  It is therefore only meaningful when
/// `prefix` is empty (the shed path); with a non-empty prefix the request line
/// is already behind it, so `is_head` is silently `false` and callers must not
/// rely on it.
fn drain_request_tracked(
    stream: &mut TcpStream,
    prefix: &[u8],
    budget: Duration,
    tail_len: &mut usize,
    wants_close: &mut bool,
    is_head: &mut bool,
    // Whether `prefix` begins with the request line.  Inferring it was a real
    // bug: the keep-alive path passes everything *after* the request line's
    // `\n`, so both the `is_head` probe below and the `http11` derivation at
    // the end silently read a *header* line, pinning `wants_close` to true and
    // disabling keep-alive on every backend but the epoll reactor.
    prefix_has_request_line: bool,
    // The version, when the caller has already parsed the request line.
    http11_hint: Option<bool>,
    // Length of the request line the caller already consumed (0 when `prefix`
    // still contains it).  `MAX_HEADER` is defined over the **whole** request
    // head — that is what `HeaderScan::scan` enforces and the scanner test pins —
    // so the thread path, which starts reading after the request line, must add
    // it back or its limit is up to `MAX_REQUEST_LINE` bytes more permissive.
    request_line_len: usize,
) -> DrainOutcome {
    let deadline = Instant::now() + budget;
    // Budget for the bytes this function may still read.
    let header_budget = MAX_HEADER.saturating_sub(request_line_len);
    // 4 KiB covers the common request header block on the stack; larger
    // blocks (cookies, long URLs) spill to the heap once and are bounded by
    // `MAX_HEADER` below.
    let mut collected: SmallVec<[u8; 4096]> = SmallVec::new();
    collected.extend_from_slice(prefix);
    // One canonical terminator/limit policy, shared with the reactor and
    // PendingConn.  `scan_after_request_line` also accepts the single blank
    // line of an empty HTTP/1.0 header block.
    let scan = HeaderScan::new();
    let mut head_end = match scan.scan_after_request_line(&collected) {
        HeaderProgress::Complete(end) => Some(end),
        _ => None,
    };
    let mut hit_cap = false;
    let mut buf = [0u8; 4096];

    while head_end.is_none() {
        if Instant::now() >= deadline {
            break;
        }
        if collected.len() >= header_budget {
            hit_cap = true;
            break;
        }
        // Never read past the header cap; a terminator arriving
        // beyond it must not be accepted as a valid header block.
        let want = (header_budget - collected.len()).min(buf.len());
        if want == 0 {
            hit_cap = true;
            break;
        }
        if arm_read_timeout_until(stream, deadline).is_err() {
            // A failed timeout install must not close with unread bytes.
            // `drain_until_eof` is non-recursive and still attempts a short
            // best-effort drain with a direct timeout.
            drain_until_eof(stream, Duration::from_millis(100), false);
            break;
        }
        match stream.read(&mut buf[..want]) {
            Ok(0) => break,
            Ok(n) => {
                collected.extend_from_slice(&buf[..n]);
                match scan.scan_after_request_line(&collected) {
                    HeaderProgress::Complete(end) => head_end = Some(end),
                    HeaderProgress::TooLargeHeader | HeaderProgress::TooLargeLine => {
                        return DrainOutcome::TooLarge;
                    }
                    HeaderProgress::Incomplete => {}
                }
            }
            Err(ref e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                break;
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }

    let Some(head_end) = head_end.filter(|&end| end <= header_budget) else {
        // Absent and incomplete are silent; hitting the size cap is a
        // protocol-visible 431, matching the reactor's `TooLargeHeader`.
        return if hit_cap {
            DrainOutcome::TooLarge
        } else {
            DrainOutcome::NoRequest
        };
    };
    if prefix_has_request_line {
        *is_head = request_line_is_head(&collected);
    }

    // Bytes past the terminator arrived in the same read as the headers and
    // count toward the body, then toward a pipelined request.
    let framing = scan_framing(&collected[..head_end], prefix_has_request_line);
    let already_read = collected.len() - head_end;
    // Any body at all (declared, chunked, or unframable) means this path cannot
    // reuse the connection for a following request.
    // One derivation for "must we drain?" *and* "may we reuse?" — see
    // `HeaderFraming::decide`.  The version is the request line's last token; an
    // HTTP/1.0 request gets no implicit keep-alive, which is what the reactor
    // already enforced and this path used to miss.
    // Only meaningful when the request line is in `collected`; otherwise the
    // caller (which parsed it) supplies the answer.
    let http11 = http11_hint.unwrap_or_else(|| {
        collected
            .split(|&b| b == b'\n')
            .next()
            .is_some_and(|line| line.trim_ascii_end().ends_with(b"HTTP/1.1"))
    });
    let decision = framing.decide(already_read, http11);
    *wants_close = !decision.keep_alive_allowed;
    if decision.body_remaining.is_none() {
        // Chunked/unsupported framing, conflicting or malformed
        // `Content-Length`, or a body over the cap: request framing cannot be
        // established, so `Complete` is only safe on peer EOF.  A budget
        // cutoff is `HeadersOnly` (half-close + linger), never a false
        // `Complete` that would RST away the just-written response .
        *tail_len = 0;
        return drain_unframed(stream, deadline);
    }
    let body_len = framing.content_length.unwrap_or(0);
    // RFC 7231 §5.1.1 permits an immediate final response without inviting
    // the body; the reactor already does this for every request.
    // Surplus bytes past the body belong to a pipelined next request; expose
    // them so the keep-alive loop can force a close .
    *tail_len = already_read.saturating_sub(body_len);
    let mut remaining = body_len.saturating_sub(already_read);
    while remaining > 0 {
        if Instant::now() >= deadline {
            break;
        }
        let want = remaining.min(buf.len());
        if arm_read_timeout_until(stream, deadline).is_err() {
            break;
        }
        match stream.read(&mut buf[..want]) {
            Ok(0) => break,
            Ok(n) => remaining -= n,
            Err(ref e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                break;
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
    let body_complete = remaining == 0;
    if !body_complete {
        return DrainOutcome::HeadersOnly;
    }

    // residual drain moved to `drain_residual`, called *after* the response
    // is written: draining before the write inserted a 20 ms sleep between
    // parsing and replying for every client (they are waiting for the
    // response, not pipelining).
    DrainOutcome::Complete
}

/// Consume inbound bytes until EOF or `budget` expires.  `half_close` sends
/// FIN first (the response is already written) and closes both directions
/// afterwards; the non-half-close form is the residual pipelined drain.
fn drain_until_eof(stream: &mut TcpStream, budget: Duration, half_close: bool) {
    if half_close {
        swallow!(stream.shutdown(Shutdown::Write));
    }
    swallow!(stream.set_read_timeout(Some(budget)));
    let deadline = Instant::now() + budget;
    let mut buf = [0u8; 4096];
    while Instant::now() < deadline {
        if arm_read_timeout_until(stream, deadline).is_err() {
            break;
        }
        match stream.read(&mut buf) {
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
    }
    if half_close {
        swallow!(stream.shutdown(Shutdown::Both));
    }
}

/// Consume late/pipelined inbound bytes after the response is on the wire, so
/// the linger overlaps client-side response delivery.
fn drain_residual(stream: &mut TcpStream) {
    drain_until_eof(stream, Duration::from_millis(20), false);
}

/// Short first-read timeout for [`serve_page`]: distinguishes a browser
/// preconnect (silent socket) from a real request cheaply, without spending
/// the full drain timeout of the accept budget on a socket that will never
/// send anything.  500 ms keeps the preconnect check cheap while leaving
/// enough headroom for a cold browser under load (the old 250 ms was prone
/// to flakes on a busy CI box).
const FIRST_BYTE_READ_TIMEOUT: Duration = Duration::from_millis(500);

/// The text content of a `<script>…</script>` constant (the exact bytes CSP
/// hashes).
fn script_content(src: &'static str) -> &'static str {
    src.strip_prefix("<script>")
        .and_then(|s| s.strip_suffix("</script>"))
        .unwrap_or(src)
}

/// CSP `script-src` built from SHA-256 hashes of the two inline scripts.
///
/// The scripts are compile-time constants, so the digest is computed once and
/// `'unsafe-inline'` can be dropped entirely — a future escaping bug in the
/// page body can no longer execute script. Hashes are computed from the
/// constants at runtime, so editing a script cannot silently desynchronise
/// them.
fn csp_script_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| {
        use base64::Engine as _;
        use sha2::{Digest, Sha256};
        let hash = |js: &str| {
            let digest = Sha256::digest(js.as_bytes());
            base64::engine::general_purpose::STANDARD.encode(digest)
        };
        format!(
            "'sha256-{}' 'sha256-{}'",
            hash(script_content(THEME_BOOTSTRAP)),
            hash(script_content(THEME_WIRE)),
        )
    })
}

/// : CSP `style-src` built from SHA-256 hashes of every page style variant.
///
/// `page_head` interleaves the shared dark/light stylesheet, the base layout,
/// and one of a closed set of `extra_css` literals.  Hashing each variant and
/// listing all digests lets `'unsafe-inline'` be dropped from `style-src` too.
/// The hashes are computed from the same constants the page embeds, so a CSS
/// edit cannot desynchronise them (the page would simply render unstyled).
fn csp_style_src() -> &'static str {
    static SRC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SRC.get_or_init(|| {
        use base64::Engine as _;
        use sha2::{Digest, Sha256};
        let css = crate::diagnostics::emitter::stylesheet::dark_css();
        let hash = |extra: &str| {
            let digest = Sha256::digest(page_style(css, extra).as_bytes());
            base64::engine::general_purpose::STANDARD.encode(digest)
        };
        let mut out = String::new();
        // The closed set of `page_head(_, extra_css)` variants — named once so
        // `page_head` can assert against the same list (see `KNOWN_EXTRA_CSS`).
        for extra in KNOWN_EXTRA_CSS {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str("'sha256-");
            out.push_str(&hash(extra));
            out.push('\'');
        }
        out
    })
}

/// Render an HTTP response head (). The only variable field is
/// `Content-Length`, so cached pages pre-render their head once and the hot
/// path writes it as a refcounted `Arc<str>` with no per-request formatting.
fn render_head(status: &str, extra_headers: &str, body_len: usize) -> String {
    format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: text/html; charset=utf-8\r\n\
         X-Content-Type-Options: nosniff\r\n\
         Cache-Control: no-store\r\n\
         Referrer-Policy: no-referrer\r\n\
         Content-Security-Policy: default-src 'none'; frame-ancestors 'none'; style-src {}; script-src {}\r\n\
         {extra_headers}\
         Content-Length: {body_len}\r\n",
        csp_style_src(),
        csp_script_src(),
    )
}

fn http_date_header() -> std::sync::Arc<str> {
    static CACHE: std::sync::OnceLock<std::sync::RwLock<(u64, std::sync::Arc<str>)>> =
        std::sync::OnceLock::new();
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let cache = CACHE.get_or_init(|| std::sync::RwLock::new((0, std::sync::Arc::from(""))));
    if let Ok(guard) = cache.read()
        && guard.0 == secs
        && !guard.1.is_empty()
    {
        return std::sync::Arc::clone(&guard.1);
    }
    let text: std::sync::Arc<str> =
        std::sync::Arc::from(format!("Date: {}\r\n", format_http_date(secs)));
    if let Ok(mut guard) = cache.write() {
        *guard = (secs, std::sync::Arc::clone(&text));
    }
    text
}

fn format_http_date(secs: u64) -> String {
    const WEEKDAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (hour, minute, second) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Howard Hinnant's civil-from-days, epoch 1970-01-01.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    if month <= 2 {
        year += 1;
    }
    let weekday = (((days % 7 + 7) % 7) + 4) % 7;
    format!(
        "{}, {:02} {} {} {:02}:{:02}:{:02} GMT",
        WEEKDAYS[weekday as usize],
        day,
        MONTHS[(month - 1) as usize],
        year,
        hour,
        minute,
        second
    )
}

fn write_header_too_large(stream: &mut TcpStream) -> std::io::Result<()> {
    let page = too_large_page();
    write_prepared_conn(stream, page.head_parts(), "", false)?;
    stream.shutdown(Shutdown::Write)
}

/// Write an HTTP response header and body with one vectored write.
///
/// `write_vectored` can accept only a prefix of the slices; loop and advance
/// until both are fully written.  This avoids the second `write_all`
/// syscall of the previous header-then-body pair while keeping the body out
/// of a combined allocation.
fn write_all_vectored<const N: usize>(
    stream: &mut TcpStream,
    mut slices: [IoSlice<'_>; N],
) -> std::io::Result<()> {
    let mut bufs: &mut [IoSlice<'_>] = &mut slices;
    while !bufs.is_empty() {
        let n = match stream.write_vectored(bufs) {
            Ok(n) => n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        if n == 0 {
            return Err(std::io::Error::from(std::io::ErrorKind::WriteZero));
        }
        IoSlice::advance_slices(&mut bufs, n);
    }
    Ok(())
}

/// Write a complete one-shot HTTP response (head rendered per call site; the
/// one-shot server serves exactly one request, so that is free).
fn write_response(
    stream: &mut TcpStream,
    status: &str,
    extra_headers: &str,
    body: &str,
) -> std::io::Result<()> {
    let header = render_head(status, extra_headers, body.len());
    // Every one-shot response closes the connection, so it must
    // say so; route through the same writer the cached pages use.
    write_prepared_conn(stream, [&header, "", ""], body, false)
}

/// Write a pre-rendered head and a body (persistent-server hot path, ).
fn write_prepared(stream: &mut TcpStream, head: &str, body: &str) -> std::io::Result<()> {
    write_prepared_conn(stream, [head, "", ""], body, false)
}

/// Cached-page writer for a split head (prefix, shared style, suffix).
fn write_prepared_parts(
    stream: &mut TcpStream,
    head: [&str; 3],
    body: &str,
) -> std::io::Result<()> {
    write_prepared_conn(stream, head, body, false)
}

/// Write a cached head + body with the connection header chosen per request.
///
/// /the cached head no longer ends with `Connection: close`; the
/// header is emitted here so an HTTP/1.1 keep-alive response does not
/// advertise a close it does not perform (and a closing response still says
/// so).  Still one vectored write.
fn write_prepared_conn(
    stream: &mut TcpStream,
    head: [&str; 3],
    body: &str,
    keep_alive: bool,
) -> std::io::Result<()> {
    const CLOSE: &str = "Connection: close\r\n\r\n";
    const KEEP: &str = "Connection: keep-alive\r\n\r\n";
    let conn = if keep_alive { KEEP } else { CLOSE };
    let date = http_date_header();
    write_all_vectored(
        stream,
        [
            IoSlice::new(head[0].as_bytes()),
            IoSlice::new(head[1].as_bytes()),
            IoSlice::new(head[2].as_bytes()),
            IoSlice::new(date.as_bytes()),
            IoSlice::new(conn.as_bytes()),
            IoSlice::new(body.as_bytes()),
        ],
    )?;
    stream.flush()
}

/// Cheap prefix check for the one-shot path : a connection that will
/// never carry an HTTP request must not consume ~500 ms of the accept budget
/// per garbage socket. HTTP methods are case-sensitive tokens.
fn looks_like_http_request(bytes: &[u8]) -> bool {
    fn is_token_byte(b: u8) -> bool {
        b.is_ascii_alphanumeric()
            || matches!(
                b,
                b'!' | b'#'
                    | b'$'
                    | b'%'
                    | b'&'
                    | b'\''
                    | b'*'
                    | b'+'
                    | b'-'
                    | b'.'
                    | b'^'
                    | b'_'
                    | b'`'
                    | b'|'
                    | b'~'
            )
    }
    if bytes.is_empty() {
        return false;
    }
    for (i, &b) in bytes.iter().enumerate() {
        // Same separator class the authoritative parser accepts
        // (`parse_request_line` splits on ASCII whitespace), so a tab-separated
        // request line cannot be rejected here and accepted by the persistent
        // backend.  The prefilter must only ever be *weaker* than the parser,
        // never stricter.  CR/LF stay excluded: a request line cannot be
        // newline-terminated before it has a target.
        if matches!(b, b' ' | b'\t') {
            return i > 0;
        }
        if !is_token_byte(b) {
            return false;
        }
    }
    // A partial token (the first read may stop mid-method) is still a
    // plausible request prefix; the next read completes it.
    true
}

#[cfg(test)]
#[test]
fn extension_method_probe_is_accepted() {
    assert!(looks_like_http_request(b"PROPFIND /E030 HTTP/1.1\r\n"));
    assert!(looks_like_http_request(b"MKCOL "));
    assert!(looks_like_http_request(b"REP"));
    assert!(!looks_like_http_request(b"\x16\x03\x01"));
    assert!(!looks_like_http_request(b" GET /"));
    // Backend parity: the persistent parser splits the request line on ASCII
    // whitespace, so the prefilter must accept a tab separator too — otherwise
    // one backend answers and the other rejects before parsing.
    assert!(looks_like_http_request(b"GET\t/E030\tHTTP/1.1\r\n"));
    // ... but CR/LF are not separators: a newline before the target means the
    // request line is malformed, not merely partial.
    assert!(!looks_like_http_request(b"GET\r"));
}

#[cfg(test)]
/// Extract the request target from a (possibly partial) request line.
fn request_target(bytes: &[u8]) -> Option<&str> {
    let line = bytes.split(|&b| b == b'\n').next()?;
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    let text = simdutf8::basic::from_utf8(line).ok()?;
    text.split_ascii_whitespace().nth(1)
}

/// Request target of a request line that is *complete* in `bytes`.
///
/// The route check must not act on a truncated first read: a legitimate
/// request line longer than the 512-byte probe (`?` query strings) would
/// otherwise compare a truncated path, answer 404 and keep waiting for a
/// request that already arrived.  Returns `None` when the line has no `\n`
/// yet, so the caller falls through to `drain_request`, which re-reads the
/// rest of the line.
#[cfg(test)]
fn complete_request_target(bytes: &[u8]) -> Option<&str> {
    memchr::memchr(b'\n', bytes)?;
    request_target(bytes)
}

/// Serve one HTTP request on `stream`.
///
/// Returns `Ok(true)` when a request was received and the page written, and
/// `Ok(false)` when the connection carried no request at all (a browser
/// speculative preconnect, garbage bytes, or a client that stalled before
/// sending).  Such a socket is closed without a response — an unsolicited
/// HTTP response would desynchronise the browser — and the caller keeps
/// waiting for the socket that carries the real request.
#[cfg(test)]
fn serve_page(stream: &mut TcpStream, body: &str) -> std::io::Result<bool> {
    serve_page_expected(stream, body, None)
}

/// [`serve_page`] with an optional expected request path . When the first
/// request on the one-shot server targets something else (a favicon probe, a
/// parallel `curl`), answer it with a 404 and keep the accept loop waiting for
/// the real page request instead of letting it consume the single page.
fn serve_page_expected(
    stream: &mut TcpStream,
    body: &str,
    expected_path: Option<&str>,
) -> std::io::Result<bool> {
    // Accepted sockets may inherit the listener's nonblocking flag on some
    // platforms; force blocking with a read timeout so draining the request
    // cannot hang the server on a silent client.  Every mode-setting call must
    // be observable: if a timeout cannot be installed, a blocking I/O could
    // wait forever, so surface the failure instead of swallowing it.
    stream.set_nonblocking(false)?;
    // Responses go out in one vectored write today; disabling Nagle removes a
    // future "why is there a 40 ms gap" class of bug if the response grows
    // past one segment.
    swallow!(stream.set_nodelay(true));
    // without SO_SNDTIMEO a client that never reads blocks
    // `write_all_vectored` forever — the only unbounded wait in the file.
    stream.set_write_timeout(Some(Duration::from_secs(1)))?;
    // A browser preconnect opens a socket and sends nothing for a while; use a
    // SHORT timeout for the first read only: silence past it almost certainly
    // means preconnect.
    stream.set_read_timeout(Some(FIRST_BYTE_READ_TIMEOUT))?;
    let mut first = [0u8; 512];
    let first_n = loop {
        match stream.read(&mut first) {
            Ok(n) => break n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            // Timed out / reset while waiting for the first byte — a preconnect.
            Err(ref e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::ConnectionReset
                        | std::io::ErrorKind::ConnectionAborted
                        | std::io::ErrorKind::UnexpectedEof
                ) =>
            {
                swallow!(stream.shutdown(Shutdown::Both));
                return Ok(false);
            }
            Err(e) => return Err(e),
        }
    };
    if first_n == 0 {
        // Peer closed without sending anything — same as a preconnect.
        swallow!(stream.shutdown(Shutdown::Both));
        return Ok(false);
    }
    // fail fast on bytes that cannot begin an HTTP request line.
    if !looks_like_http_request(&first[..first_n]) {
        swallow!(stream.shutdown(Shutdown::Both));
        return Ok(false);
    }
    serve_page_after_prefix(stream, body, expected_path, &first[..first_n], None)
}

/// Post-probe serving shared by the blocking non-unix/test path and the
/// non-blocking one-shot state machine.  `first` is the bytes already read
/// (request line and possibly more); the socket may still be non-blocking, so
/// this forces blocking mode with bounded timeouts before draining.
fn serve_page_after_prefix(
    stream: &mut TcpStream,
    body: &str,
    expected_path: Option<&str>,
    first: &[u8],
    head: Option<RequestHead>,
) -> std::io::Result<bool> {
    stream.set_nonblocking(false)?;
    swallow!(stream.set_nodelay(true));
    stream.set_write_timeout(Some(Duration::from_secs(1)))?;
    stream.set_read_timeout(Some(Duration::from_secs(1)))?;
    let (line, leftover, pre_drain) = if let Some(head) = head {
        let end = head.head_end.min(first.len());
        let mut leftover: SmallVec<[u8; 512]> = SmallVec::new();
        leftover.extend_from_slice(&first[end..]);
        let drain = drain_after_head(stream, head.framing, leftover.len(), Duration::from_secs(1));
        (head.line, leftover, Some(drain))
    } else if expected_path.is_some() {
        let mut too_large = false;
        let mut malformed = false;
        match read_request_line(
            stream,
            first,
            Duration::from_secs(2),
            &mut too_large,
            &mut malformed,
        ) {
            Some((line, leftover)) => (line, leftover, None),
            None => {
                if too_large {
                    swallow!(write_header_too_large(stream));
                    linger_after_half_close(stream);
                } else if malformed {
                    let resp = bad_request_response();
                    swallow!(write_prepared_parts(stream, resp.head_parts(), resp.body()));
                    linger_after_half_close(stream);
                }
                swallow!(stream.shutdown(Shutdown::Both));
                return Ok(false);
            }
        }
    } else {
        // Test-only `serve_page`: the whole buffer is still here, request line
        // included, so drain it with the matching flag.
        let drain = drain_whole_head(stream, first, Duration::from_secs(1));
        (
            CompactString::new(""),
            SmallVec::from_slice(first),
            Some(drain),
        )
    };
    let prefix: &[u8] = &leftover;

    if expected_path.is_some() {
        // HEAD is decided from the raw line *before* validation: every 400 branch
        // below (unparseable line, bad version, bad target) must suppress the body
        // too, and `HEAD /E030 HTTP/2` used to receive one.  `space or tab` after
        // the method, matching `parse_request_line`'s separator policy.
        let head_only = line
            .as_bytes()
            .strip_prefix(b"HEAD")
            .is_some_and(|rest| rest.first().is_some_and(|&b| b == b' ' || b == b'\t'));
        // Validate the full three-token grammar and version before
        // any routing decision, exactly like the persistent dispatcher.
        let Some(req) = parse_request_line(line.as_ref()) else {
            let drain = drain_or_pre(stream, prefix, pre_drain, Duration::from_secs(1));
            let resp = bad_request_response();
            let resp = if head_only { resp.as_head() } else { resp };
            return finish_response(stream, drain, &resp, false);
        };
        if !matches!(req.version, "HTTP/1.0" | "HTTP/1.1") {
            let drain = drain_or_pre(stream, prefix, pre_drain, Duration::from_secs(1));
            let resp = bad_request_response();
            let resp = if head_only { resp.as_head() } else { resp };
            return finish_response(stream, drain, &resp, false);
        }
        // Origin-form target is required; `GET E030 HTTP/1.1` must not be
        // served by the one-shot while the persistent router rejects it.
        if !req.target.starts_with('/') {
            let drain = drain_or_pre(stream, prefix, pre_drain, Duration::from_secs(1));
            let resp = bad_request_response();
            let resp = if head_only { resp.as_head() } else { resp };
            return finish_response(stream, drain, &resp, false);
        }
        // A non-GET/HEAD must not receive a full body (protocol violation); answer
        // 405 with `Allow: GET, HEAD` and keep waiting for the real page request.
        if req.method != "GET" && req.method != "HEAD" {
            // The 1 s read/write timeouts are armed once at the top of this
            // function; re-arming them per response was two `setsockopt`s (and
            // two `?` paths that closed the connection for a no-op change).
            let drain = drain_or_pre(stream, prefix, pre_drain, Duration::from_secs(1));
            let resp = method_not_allowed_response();
            return finish_response(stream, drain, &resp, false);
        }
        if let Some(expected) = expected_path {
            let target = req.target;
            let expected_code = expected.trim_matches('/');
            let raw = target.split('?').next().unwrap_or("/");
            let alt = format!("{expected}/");
            let canonical = canonical_path_code(target);
            let matches = canonical.as_deref() == Some(expected_code)
                || raw == expected
                || raw == alt
                // A browser's first hit on a one-shot server is almost always `/` — the
                // server was started for exactly one page, so serving it on the root is
                // the whole point.  It used to 404 the first click and then sit out the
                // remaining accept budget.  (See-also navigation is unaffected: those
                // targets carry a code and are handled below.)
                || raw == "/";
            if !matches {
                // See-also navigation: a registered code is a real page even
                // when it is not this one-shot's initial target.  Only an
                // unknown code reaches the 404 branch below.
                if let Some(code) = canonical.as_deref()
                    && let Some(page) = explain_page_cached(code)
                {
                    let drain = drain_or_pre(stream, prefix, pre_drain, Duration::from_secs(1));
                    if head_only {
                        return finish_head(
                            stream,
                            drain,
                            |stream| write_prepared_parts(stream, page.head_parts(), ""),
                            true,
                        );
                    }
                    match drain {
                        DrainOutcome::NoRequest => {
                            swallow!(stream.shutdown(Shutdown::Both));
                            return Ok(false);
                        }
                        DrainOutcome::HeadersOnly => {
                            write_prepared_parts(stream, page.head_parts(), &page.body)?;
                            linger_after_half_close_for(stream, ONE_SHOT_LINGER);
                        }
                        DrainOutcome::Complete => {
                            write_prepared_parts(stream, page.head_parts(), &page.body)?;
                            drain_residual(stream);
                            swallow!(stream.shutdown(Shutdown::Both));
                        }
                        DrainOutcome::TooLarge => {
                            write_header_too_large(stream)?;
                            linger_after_half_close(stream);
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }
                // Drain the whole request before writing: closing with unread
                // inbound bytes makes the kernel send RST, which can discard
                // the 404 response (the discipline).
                let decoded = percent_decode(raw);
                let key = decoded.strip_prefix('/').unwrap_or(decoded.as_str());
                let key = if key.is_empty() { "/" } else { key };
                let page = not_found_page_cached(key);
                let drain = drain_or_pre(stream, prefix, pre_drain, Duration::from_secs(1));
                if head_only {
                    return finish_head(
                        stream,
                        drain,
                        |stream| write_prepared_parts(stream, page.head_parts(), ""),
                        false,
                    );
                }
                match drain {
                    DrainOutcome::NoRequest => {
                        swallow!(stream.shutdown(Shutdown::Both));
                    }
                    DrainOutcome::HeadersOnly => {
                        swallow!(write_prepared_parts(stream, page.head_parts(), &page.body));
                        linger_after_half_close_for(stream, ONE_SHOT_LINGER);
                    }
                    DrainOutcome::Complete => {
                        swallow!(write_prepared_parts(stream, page.head_parts(), &page.body));
                        drain_residual(stream);
                        swallow!(stream.shutdown(Shutdown::Both));
                    }
                    DrainOutcome::TooLarge => {
                        swallow!(write_header_too_large(stream));
                        linger_after_half_close(stream);
                    }
                }
                return Ok(false);
            }
        }
        // HEAD gets the identical GET head (including Content-Length) and no body.
        if head_only {
            let drain = drain_or_pre(stream, prefix, pre_drain, Duration::from_secs(1));
            let head = render_head("200 OK", "", body.len());
            return finish_head(
                stream,
                drain,
                |stream| write_prepared(stream, &head, ""),
                true,
            );
        }
    }

    match drain_or_pre(stream, prefix, pre_drain, Duration::from_secs(1)) {
        DrainOutcome::NoRequest => {
            swallow!(stream.shutdown(Shutdown::Both));
            Ok(false)
        }
        DrainOutcome::HeadersOnly => {
            // The body was not fully drained; a hard close would RST. Write the
            // response, half-close, and linger until EOF/budget .
            swallow!(stream.set_read_timeout(Some(Duration::from_millis(50))));
            write_response(stream, "200 OK", "", body)?;
            linger_after_half_close_for(stream, ONE_SHOT_LINGER);
            Ok(true)
        }
        DrainOutcome::Complete => {
            write_response(stream, "200 OK", "", body)?;
            // Residual drain after the response, overlapping the client's
            // read, instead of a 20 ms sleep before it.
            drain_residual(stream);
            // Fully drained: no unread inbound bytes, so a hard close is FIN.
            swallow!(stream.shutdown(Shutdown::Both));
            Ok(true)
        }
        DrainOutcome::TooLarge => {
            swallow!(write_header_too_large(stream));
            linger_after_half_close(stream);
            Ok(false)
        }
    }
}

/// Decode and normalize a request target to a canonical error code.
///
/// Shared policy with the persistent router : strip `?query`,
/// percent-decode, exactly one leading slash, reject a second leading slash,
/// exactly one trailing slash, then canonicalize the code spelling.  `"/"`
/// (the index) has no code and returns `None`; callers handle the index first.
fn canonical_path_code(target: &str) -> Option<&'static str> {
    let raw = target.split('?').next().unwrap_or("/");
    // Slash policy runs on the *encoded* path: `%2F` is an encoded slash and
    // must not be normalised into a path separator (otherwise `/E030%2F`
    // would alias `/E030/`).  Percent-decoding happens after the slash shape
    // is fixed, so `%45%30%33%30` still resolves to `E030`.
    let p = raw.strip_prefix('/').unwrap_or(raw);
    if p.starts_with('/') {
        // `//E030` names a different resource, not `E030`.
        return None;
    }
    let p = p.strip_suffix('/').unwrap_or(p);
    let decoded = percent_decode(p);
    let p = decoded.as_str();
    if p.is_empty() {
        return None;
    }
    ErrCode::try_new(p)
        .ok()
        .and_then(|e| e.entry().map(|entry| entry.code))
}

/// Write a head-only response after the request drain, choosing the same
/// half-close/linger policy as the GET path.  `delivered` distinguishes the
/// expected page (which completes the one-shot server) from a 404/431.
fn finish_head<F>(
    stream: &mut TcpStream,
    drain: DrainOutcome,
    write_head: F,
    delivered: bool,
) -> std::io::Result<bool>
where
    F: FnOnce(&mut TcpStream) -> std::io::Result<()>,
{
    match drain {
        DrainOutcome::NoRequest => {
            swallow!(stream.shutdown(Shutdown::Both));
            Ok(false)
        }
        DrainOutcome::HeadersOnly => {
            write_head(stream)?;
            linger_after_half_close_for(stream, ONE_SHOT_LINGER);
            Ok(delivered)
        }
        DrainOutcome::Complete => {
            write_head(stream)?;
            drain_residual(stream);
            swallow!(stream.shutdown(Shutdown::Both));
            Ok(delivered)
        }
        DrainOutcome::TooLarge => {
            write_header_too_large(stream)?;
            linger_after_half_close(stream);
            Ok(false)
        }
    }
}

/// Shared post-response policy for the blocking backends: keep-alive returns
/// immediately; a framed body drains residual bytes before FIN; an unframed
/// body lingers to EOF/budget so unread inbound bytes cannot turn the close
/// into an RST that discards the response.
fn finish_after_response(
    stream: &mut TcpStream,
    outcome: DrainOutcome,
    keep_alive: bool,
) -> std::io::Result<()> {
    if keep_alive {
        return Ok(());
    }
    if outcome == DrainOutcome::Complete {
        drain_residual(stream);
        swallow!(stream.shutdown(Shutdown::Both));
    } else {
        linger_after_half_close(stream);
    }
    Ok(())
}

fn finish_response(
    stream: &mut TcpStream,
    drain: DrainOutcome,
    response: &HttpResponse,
    delivered: bool,
) -> std::io::Result<bool> {
    match drain {
        DrainOutcome::NoRequest => {
            swallow!(stream.shutdown(Shutdown::Both));
            Ok(false)
        }
        DrainOutcome::HeadersOnly => {
            write_prepared_parts(stream, response.head_parts(), response.body())?;
            finish_after_response(stream, drain, false)?;
            Ok(delivered)
        }
        DrainOutcome::Complete => {
            write_prepared_parts(stream, response.head_parts(), response.body())?;
            finish_after_response(stream, drain, false)?;
            Ok(delivered)
        }
        DrainOutcome::TooLarge => {
            write_header_too_large(stream)?;
            linger_after_half_close(stream);
            Ok(false)
        }
    }
}

/// Half-close, then consume inbound bytes until EOF or a short budget.
///
/// A single 256-byte read is not a drain: if the client is still streaming a
/// larger body, unread bytes remain and the final close can RST the response
/// just written .
fn linger_after_half_close_for(stream: &mut TcpStream, budget: Duration) {
    drain_until_eof(stream, budget, true);
}

/// Half-close linger for a complete response whose request body was not fully
/// drained (the 431/malformed paths plus the `HeadersOnly`/shed-503 arms).
/// Shares [`ONE_SHOT_LINGER`] with the one-shot `HeadersOnly` call sites: all
/// write a complete response before draining, so they need the same
/// browser-drain window (the old 50 ms was shorter than a slow local client's
/// drain).
fn linger_after_half_close(stream: &mut TcpStream) {
    linger_after_half_close_for(stream, ONE_SHOT_LINGER);
}

/// Escape a string for HTML embedding.
///
/// Most fields on the explain page (title, code, category, explain text)
/// come from the compiler's static error-code tables and contain no `& < >
/// " '` characters.  For those the `Cow::Borrowed` fast path avoids the
/// allocation entirely — only strings that actually need escaping are
/// owned.
fn escape_html(s: &str) -> Cow<'_, str> {
    let bytes = s.as_bytes();
    // Single fused pass: collect each escape site and its class count at the
    // same time, so the reservation is exact without a second full scan.
    let mut positions: crate::smallvec::SmallVec<[usize; 16]> = crate::smallvec::SmallVec::new();
    let (mut amp, mut lt, mut gt, mut quot, mut apos) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for idx in super::renderers::html::HtmlEscapeIter::new(bytes) {
        match bytes[idx] {
            b'&' => amp += 1,
            b'<' => lt += 1,
            b'>' => gt += 1,
            b'"' => quot += 1,
            b'\'' => apos += 1,
            // The merged iterator only yields those five bytes.
            _ => debug_assert!(false, "HtmlEscapeIter yielded a non-escape byte"),
        }
        positions.push(idx);
    }
    if positions.is_empty() {
        return Cow::Borrowed(s);
    }
    // Saturating arithmetic: `s.len() + extra` can overflow `usize` for an
    // adversarial 32-bit input dominated by `&`.  A too-small reservation is
    // harmless (`String` grows); wrapping to a tiny value is not.
    let extra = amp
        .saturating_mul(4)
        .saturating_add(lt.saturating_mul(3))
        .saturating_add(gt.saturating_mul(3))
        .saturating_add(quot.saturating_mul(5))
        .saturating_add(apos.saturating_mul(4));
    let mut out = String::with_capacity(s.len().saturating_add(extra));
    let mut start = 0;
    for idx in positions {
        out.push_str(&s[start..idx]);
        match bytes[idx] {
            b'&' => out.push_str("&amp;"),
            b'<' => out.push_str("&lt;"),
            b'>' => out.push_str("&gt;"),
            b'"' => out.push_str("&quot;"),
            b'\'' => out.push_str("&#39;"),
            // Keep the release path total — a panic inside an HTTP handler
            // would take down the connection thread (and, under
            // `thread::scope`, the whole persistent server).
            _ => {
                debug_assert!(false, "HtmlEscapeIter yielded a non-escape byte");
                if let Some(ch) = s[idx..].chars().next() {
                    // Defence in depth: never copy an unexpected byte raw
                    // into HTML, even if the merged iterator's contract grows.
                    out.push_str("&#");
                    out.push_str(&(ch as u32).to_string());
                    out.push(';');
                    start = idx + ch.len_utf8();
                    continue;
                }
            }
        }
        start = idx + 1;
    }
    out.push_str(&s[start..]);
    Cow::Owned(out)
}

/// Theme bootstrap shared by the explain page, index, 404 and status pages.
///
/// `THEME_BOOTSTRAP` runs synchronously from `<head>` and sets the class on
/// `document.documentElement` *before* the body is parsed, so a light-mode
/// user never sees a dark first paint . `THEME_WIRE` only wires the
/// toggle and runs once the DOM exists.
const THEME_BOOTSTRAP: &str = r#"<script>
(function () {
  var t = localStorage.getItem('ponent-theme');
  if (t === 'light' || (!t && matchMedia('(prefers-color-scheme: light)').matches)) {
    document.documentElement.classList.add('light');
  }
})();
</script>"#;

const THEME_WIRE: &str = r#"<script>
(function () {
  function wire() {
    var button = document.getElementById('theme-toggle');
    if (button) {
      button.addEventListener('click', function () {
        var light = document.documentElement.classList.toggle('light');
        localStorage.setItem('ponent-theme', light ? 'light' : 'dark');
      });
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wire);
  } else {
    wire();
  }
})();
</script>"#;

const EXPLAIN_CSS: &str = r#"  .code { color: var(--accent); font-weight: bold; font-size: 1.2em; }
  .title { color: var(--fg); font-size: 1.1em; }
  .category { color: var(--muted); margin-bottom: 20px; }
  .explain { background: var(--diag-bg); padding: 20px; border-radius: 8px; line-height: 1.6; white-space: pre-wrap; color: var(--explain-fg); }
  .url { color: var(--muted); margin-top: 20px; font-size: 0.85em; }
  .see-also { margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--fg); }
  .see-also h3 { color: var(--accent); font-size: 1em; margin: 0 0 8px 0; }
  .see-also ul { list-style: none; padding: 0; margin: 0; }
  .see-also li { margin: 4px 0; }
  .see-also .code { font-weight: bold; }"#;

const INDEX_CSS: &str = r#"  h2 { color: var(--accent); margin-top: 28px; }
  ul { list-style: none; padding: 0; }
  li { margin: 4px 0; }
  a { text-decoration: none; }
  a:hover { text-decoration: underline; }
  .code { font-weight: bold; }
  .title { color: var(--muted); }"#;

/// The shared base layout wrapped around every page's stylesheet.
///
/// Extracted from `page_head` so `csp_style_src` can hash exactly the text the
/// browser sees (); both paths call `page_style`, so they cannot drift.
const BASE_STYLE: &str = "  body { font-family: 'DejaVu Sans Mono', 'Consolas', monospace; background: var(--bg); color: var(--fg); padding: 40px; max-width: 900px; margin: auto; line-height: 1.55; }\n  #theme-toggle { position: fixed; top: 16px; right: 16px; background: var(--bg); color: var(--fg); border: 1px solid var(--fg); border-radius: 6px; padding: 6px 14px; cursor: pointer; font-family: inherit; font-size: 0.85em; }\n  #theme-toggle:hover { opacity: 0.8; }\n  a { color: var(--accent); }\n  h1 { color: var(--error-border); }\n";

/// Exact `<style>` element content for a page.
fn page_style(css: &str, extra_css: &str) -> String {
    format!("\n{css}\n{BASE_STYLE}{extra_css}\n")
}

/// Shared HTML head: the process-wide dark/light stylesheet, viewport meta,
/// base layout, and the theme toggle.  `extra_css` is trusted literal CSS
/// (never user input); `title` is escaped.
/// The closed set of page styles this server can serve.
///
/// A new variant must be added here *and* is asserted in `page_head`: passing an
/// unrhashed style compiles, ships, and renders the page unstyled (the browser
/// rejects the `<style>` block on a hash mismatch) with no server-side signal.
const KNOWN_EXTRA_CSS: [&str; 3] = [EXPLAIN_CSS, INDEX_CSS, ""];

fn page_head(title: &str, extra_css: &str) -> String {
    debug_assert!(
        KNOWN_EXTRA_CSS.contains(&extra_css),
        "extra_css is not in csp_style_src's hashed set — the browser would \
         reject the <style> block; add it to KNOWN_EXTRA_CSS"
    );
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{style}</style>
{THEME_BOOTSTRAP}
{THEME_WIRE}
</head>
<body>
<button id="theme-toggle" type="button">Toggle theme</button>
"#,
        title = escape_html(title),
        style = page_style(
            crate::diagnostics::emitter::stylesheet::dark_css(),
            extra_css
        ),
    )
}

/// Minimal complete page for a protocol-level status (400/405).  `heading`
/// is escaped; `detail_html` is trusted literal markup.
fn simple_page(title: &str, heading: &str, detail_html: &str) -> String {
    let mut out = page_head(title, "");
    swrite!(
        out,
        "<h1>{}</h1>\n<p>{}</p>\n<p><a href=\"/\">Back to the index</a></p>\n</body>\n</html>\n",
        escape_html(heading),
        detail_html,
    );
    out
}

fn build_explain_page(ec: &ErrCode) -> String {
    let title = escape_html(ec.title());
    let code = escape_html(ec.code());
    let category = escape_html(ec.category().label());
    let explain = escape_html(ec.explain());
    // `url()` owns the result, so it must outlive the borrow that
    // `escape_html` makes into it.
    let url_owned = ec.url();
    let url = escape_html(&url_owned);

    // Curated or category-sibling related codes for a "see also"
    // section — the same source of truth used by the terminal renderer.
    let explicit = crate::diagnostics::error_code::related_codes_of(ec.code());
    let related: &[&'static str] = if explicit.is_empty() {
        crate::diagnostics::error_code::category_siblings_of(ec.code(), 4)
    } else {
        explicit
    };
    let see_also = if related.is_empty() {
        String::new()
    } else {
        let mut s = String::from("<div class=\"see-also\"><h3>See also</h3><ul>\n");
        for rel in related {
            let title = crate::diagnostics::error_code::lookup(rel)
                .map(|e| e.title)
                .unwrap_or("(unknown)");
            swrite!(
                s,
                "<li><a href=\"/{code}\"><span class=\"code\">{code}</span></a> — {title}</li>\n",
                code = escape_html(rel),
                title = escape_html(title),
            );
        }
        s.push_str("</ul></div>\n");
        s
    };

    let mut out = page_head(
        &format!("{}: {} — Ponent", ec.code(), ec.title()),
        EXPLAIN_CSS,
    );
    swrite!(
        out,
        "<h1>{code}</h1>\n\
         <div class=\"code\">{code}</div>\n\
         <div class=\"title\">{title}</div>\n\
         <div class=\"category\">{category}</div>\n\
         <div class=\"explain\">{explain}</div>\n\
         {see_also}<div class=\"url\"><a href=\"{url}\">{url}</a></div>\n\
         </body>\n</html>\n",
    );
    out
}

/// Maximum connections handled concurrently by
/// [`serve_explain_persistent`].  Browsers open several parallel connections
/// (the request plus speculative preconnects), so a serialized accept loop
/// makes each preconnect pay the full read timeout of the connection ahead of
/// it; but the cap stays finite so a hostile local process cannot exhaust
/// threads by opening sockets in a tight loop.
const MAX_CONCURRENT_CONNECTIONS: usize = 8;

/// Body returned to a connection shed at the concurrency cap . A 503 with
/// `Retry-After` lets a browser recover; a silent close surfaces as
/// `ERR_EMPTY_RESPONSE`.
const SHED_BODY: &str = "<!DOCTYPE html><html><head><title>503 — busy</title></head>\
<body><h1>503 — busy</h1><p>Too many concurrent connections; retry shortly.</p></body></html>";

/// Pre-rendered 503 for the shed path.  The overload path must not
/// pay `format!` + CSP lookups; it is the one response that runs under load.
fn shed_page() -> &'static CachedPage {
    static PAGE: std::sync::OnceLock<CachedPage> = std::sync::OnceLock::new();
    PAGE.get_or_init(|| {
        CachedPage::build(
            "503 Service Unavailable",
            "Retry-After: 1\r\n",
            SHED_BODY.to_string(),
        )
    })
}

fn too_large_page() -> &'static CachedPage {
    static PAGE: std::sync::OnceLock<CachedPage> = std::sync::OnceLock::new();
    PAGE.get_or_init(|| CachedPage::build("431 Request Header Fields Too Large", "", String::new()))
}

/// Decrements the live-connection counter when the connection handler exits,
/// **including on panic**.  The previous `fetch_sub` after the handler call
/// was skipped by an unwind, permanently wedging the server at its cap.
struct ActiveConnectionGuard(std::sync::Arc<AtomicUsize>);

impl Drop for ActiveConnectionGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Serve the explain index and per-code pages until the process is
/// killed (Ctrl+C terminates the process; no explicit signal handling is
/// installed).
///
/// Unlike [`serve_explain`] (which waits for one browser connection,
/// serves one page, and returns), this blocks forever.  Requests are
/// dispatched by path:
///
/// - `GET /` → index page listing every error code, grouped by category
/// - `GET /<code>` → that code's explanation page
/// - anything else → a minimal 404 page
///
/// The HTTP handling is deliberately minimal: HTTP/1.1 keep-alive is honored
/// (closed on HTTP/1.0, an explicit `Connection: close`, incomplete framing,
/// or a pipelined tail); chunked bodies are not decoded — unknown framing is
/// drained until EOF/budget and then closed.
pub fn serve_explain_persistent(listener: &TcpListener) -> std::io::Result<()> {
    serve_explain_persistent_impl(listener, None)
}

/// Bounded persistent server: exits after `idle` once at least one page has
/// been served, or with `TimedOut` if no client ever arrives.
pub fn serve_explain_persistent_bounded(
    listener: &TcpListener,
    idle: Duration,
) -> std::io::Result<()> {
    serve_explain_persistent_impl(listener, Some(idle))
}

fn serve_explain_persistent_impl(
    listener: &TcpListener,
    idle: Option<Duration>,
) -> std::io::Result<()> {
    if !listener
        .local_addr()
        .map(|a| a.ip().is_loopback())
        .unwrap_or(false)
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "explain server: listener is not bound to a loopback address",
        ));
    }
    // Preserve the caller's listener mode: the Linux reactor flips it to
    // nonblocking below, and the guard restores the original on every return.
    let _restore = RestoreNonblocking::new(listener, false)?;
    // On Linux the *only* path is the epoll reactor below, which enables
    // `TCP_DEFER_ACCEPT` in its own scope.  Enabling it here as well stacked two
    // guards on the same socket: benign (the inner guard restores to the outer's
    // value, so drop order is LIFO-correct) but it made the socket's state
    // depend on two `Drop` impls for no gain, and the second `setsockopt` was
    // pure syscall overhead.  Non-Linux keeps it here because the
    // thread-per-connection path is the one that continues below.
    #[cfg(not(target_os = "linux"))]
    let _defer_accept = enable_tcp_defer_accept(listener);
    // Thread-per-connection: a slow client (its 500 ms read-timeout path, or
    // one that stalls mid-request) must not delay accepting the browser's
    // other connections.  The shared caches (`explain_page_cached`,
    // `not_found_page_cached` are `Mutex`-guarded, `build_index_page` sits
    // behind a `OnceLock`) are already thread-safe.
    // Prewarm the page cache. Every page is a pure function of the static
    // error-code tables, so building them once at startup turns the first
    // request for any code into a pure `Arc` clone instead of a synchronous
    // build on the connection thread. ~452 pages ≈ a few ms and a few MB.
    for (_, entries) in crate::diagnostics::error_code::code_entries_by_category() {
        for entry in entries {
            swallow!(explain_page_cached(entry.code));
        }
    }
    // the index and the CSP digests are otherwise paid on the first
    // request; both are pure functions of the static tables/constants.
    swallow!(build_index_page());
    swallow!(csp_style_src());
    swallow!(csp_script_src());
    // Protocol-error pages are pure constants too; prewarm them so the first
    // malformed/overload request does not pay a build on the accept path.
    swallow!(bad_request_response());
    swallow!(method_not_allowed_response());
    swallow!(shed_page());
    swallow!(too_large_page());
    // Linux uses a single non-blocking epoll reactor: one event loop owns
    // every connection, so a slow client costs no thread and cannot delay
    // accepts.  Other platforms keep the scoped thread-per-connection path.
    // Exactly one of the two blocks below survives `cfg` expansion, and the
    // survivor IS this function's tail expression.  The old shape returned on
    // Linux and left the entire thread path statically unreachable there — the
    // `unreachable_code` lint fired over ~150 lines on the primary target, which
    // is what made the parity argument ("the thread path stays type-checked")
    // cost a warning instead of buying anything.
    #[cfg(target_os = "linux")]
    {
        serve_explain_persistent_epoll(listener, idle)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let active = std::sync::Arc::new(AtomicUsize::new(0));
        // A permanently failing listener (closed fd → `EBADF`, a device error)
        // must not leave the process spinning at 50 ms per iteration forever with
        // no diagnostic.  A *transient* error (EMFILE, ENFILE) is followed by
        // success, which resets the counter, so only a genuinely stuck listener
        // reaches the cap.
        let mut consecutive_accept_errors = 0usize;
        // H5: exponential backoff for transient fd/memory exhaustion, capped at
        // 1 s and excluded from the give-up counter.
        let mut accept_backoff_ms = 50u64;
        // a dead listener must not be reported as a successful server exit.
        let mut terminal_error: Option<std::io::Error> = None;
        if idle.is_some() {
            listener.set_nonblocking(true)?;
        }
        let first_deadline = Instant::now() + Duration::from_secs(10);
        let mut served_any = false;
        let mut last_activity = Instant::now();
        let served_count = std::sync::Arc::new(AtomicUsize::new(0));
        let mut seen_served = 0usize;
        std::thread::scope(|scope| {
            loop {
                let now = Instant::now();
                let written = served_count.load(Ordering::Relaxed);
                if written != seen_served {
                    seen_served = written;
                    served_any = true;
                    last_activity = now;
                }
                if let Some(idle) = idle {
                    if !served_any && now >= first_deadline && active.load(Ordering::Relaxed) == 0 {
                        terminal_error = Some(std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            "explain server: no client connected before the first-request deadline",
                        ));
                        break;
                    }
                    if served_any
                        && active.load(Ordering::Relaxed) == 0
                        && now.saturating_duration_since(last_activity) >= idle
                    {
                        break;
                    }
                }
                let incoming = listener.accept();
                match incoming {
                    Ok((mut stream, _)) => {
                        consecutive_accept_errors = 0;
                        accept_backoff_ms = 50;
                        // Per-connection errors must not kill the server.
                        if active.load(Ordering::Relaxed) >= MAX_CONCURRENT_CONNECTIONS {
                            // At the cap: shed with a short budget instead of serving inline.  The
                            // full handler would hold the accept loop for up to ~4 s
                            // (2 s read + 1 s drain + 1 s write), which a handful of
                            // trickle clients could force indefinitely.  A bounded
                            // drain + close is the same treatment preconnects get;
                            // browsers retry.  The accept loop stall drops to ~0.1 s.
                            // if the short read timeout cannot be installed, a blocking drain
                            // would stall the accept loop, so close immediately.
                            if stream
                                .set_read_timeout(Some(Duration::from_millis(100)))
                                .is_err()
                            {
                                swallow!(stream.shutdown(Shutdown::Both));
                                continue;
                            }
                            let (outcome, is_head) = drain_request_with_head(
                                &mut stream,
                                &[],
                                Duration::from_millis(100),
                            );
                            if outcome == DrainOutcome::NoRequest {
                                // H3: a silent preconnect never made a request;
                                // writing a 503 to it is unsolicited and can
                                // desynchronise the browser.  Close instead.
                                swallow!(stream.shutdown(Shutdown::Both));
                                continue;
                            }
                            if outcome == DrainOutcome::TooLarge {
                                swallow!(write_header_too_large(&mut stream));
                                linger_after_half_close(&mut stream);
                                continue;
                            }
                            // tell the client we are overloaded so a browser
                            // can retry, instead of an empty close.
                            let page = shed_page();
                            let body: &str = if is_head { "" } else { &page.body };
                            swallow!(write_prepared_parts(&mut stream, page.head_parts(), body));
                            swallow!(drain_residual(&mut stream));
                            if outcome == DrainOutcome::Complete {
                                swallow!(stream.shutdown(Shutdown::Both));
                            } else {
                                // Unread inbound bytes + hard close = RST, which
                                // can destroy the 503 we just wrote.  The shared
                                // linger loops to EOF/budget (H5/).
                                linger_after_half_close(&mut stream);
                            }
                            continue;
                        }
                        active.fetch_add(1, Ordering::Relaxed);
                        let guard = ActiveConnectionGuard(std::sync::Arc::clone(&active));
                        // A default thread stack is 8 MiB of *virtual* address
                        // space per connection; this handler only needs a few KiB
                        // (the largest frames are the 4 KiB drain buffer and a
                        // 4096-byte inline SmallVec).  64 KiB keeps the resident
                        // footprint honest and still leaves room for debug-build
                        // frames.  If spawn itself fails, dropping the closure
                        // drops the guard, so the counter never leaks.
                        let served_thread = std::sync::Arc::clone(&served_count);
                        let spawned = std::thread::Builder::new()
                            .name("ponent-explain".into())
                            .stack_size(128 * 1024)
                            .spawn_scoped(scope, move || {
                                let _guard = guard;
                                // H6: `thread::scope` re-raises a scoped thread's
                                // panic at the boundary, which would abort the
                                // whole persistent server.  Contain one malformed
                                // request instead; the guard still drops on
                                // unwind.
                                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    swallow!(
                                        handle_persistent_connection(&mut stream, &served_thread)
                                    );
                                }))
                                .is_err()
                                {
                                    eprintln!(
                                        "ponent explain server: connection handler panicked; \
                                     connection closed"
                                    );
                                }
                            });
                        if let Err(e) = spawned {
                            // The guard has already been dropped (closure
                            // destructuring on the failed spawn); just avoid
                            // spinning on resource exhaustion.
                            swallow!(e);
                            std::thread::sleep(Duration::from_millis(10));
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(10));
                        continue;
                    }
                    // EINTR / transient errors — retry.  Any other persistent
                    // accept failure (`EMFILE`, a closed listener, …) backs off
                    // instead of spinning: the old `continue` turned a resource
                    // exhaustion condition into a hot loop that burned a core.
                    // A short sleep keeps the server alive to retry without
                    // spinning, so a transient `EMFILE` recovers as descriptors
                    // free up rather than permanently killing the server.
                    Err(e) => {
                        // EMFILE/ENFILE/ENOBUFS/ENOMEM are transient
                        // resource exhaustion; ECONNABORTED/EPROTO are normal
                        // under a backlog RST storm.  None may permanently kill
                        // the server.  EBADF/InvalidInput still trips the cap.
                        #[cfg(unix)]
                        let transient = matches!(
                            e.raw_os_error(),
                            Some(
                                libc::EMFILE
                                    | libc::ENFILE
                                    | libc::ENOBUFS
                                    | libc::ENOMEM
                                    | libc::ECONNABORTED
                                    | libc::EPROTO
                            )
                        );
                        #[cfg(not(unix))]
                        let transient = matches!(
                            e.kind(),
                            std::io::ErrorKind::ConnectionAborted
                                | std::io::ErrorKind::ConnectionReset
                        );
                        match e.kind() {
                            std::io::ErrorKind::Interrupted => {}
                            _ if transient => {
                                accept_backoff_ms = (accept_backoff_ms * 2).min(1000);
                                std::thread::sleep(Duration::from_millis(accept_backoff_ms));
                            }
                            _ => {
                                consecutive_accept_errors += 1;
                                if consecutive_accept_errors >= MAX_CONSECUTIVE_ACCEPT_ERRORS {
                                    // The listener is not recovering (fd closed /
                                    // permanently invalid).  Surface it once and
                                    // stop rather than burning a core forever.
                                    eprintln!(
                                        "ponent explain server: giving up after \
                                     {consecutive_accept_errors} consecutive accept errors: {e}"
                                    );
                                    terminal_error = Some(e);
                                    break;
                                }
                                std::thread::sleep(Duration::from_millis(50));
                            }
                        }
                        continue;
                    }
                }
            }
        });
        match terminal_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// Linux epoll reactor for the persistent explain server.
///
/// One non-blocking event loop owns every connection: request accumulation,
/// response generation, partial writes and keep-alive reuse all take place in
/// the loop, and per-connection deadlines bound silent preconnects, partial
/// requests and stalled readers.  A slow client therefore costs no thread and
/// cannot delay accepting the browser's other connections.
#[cfg(target_os = "linux")]
fn serve_explain_persistent_epoll(
    listener: &TcpListener,
    idle: Option<Duration>,
) -> std::io::Result<()> {
    use rustc_hash::FxHashMap;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

    let _defer_accept = enable_tcp_defer_accept(listener);

    const MAX_EVENTS: usize = 64;
    const WRITE_TIMEOUT: Duration = Duration::from_secs(5);
    /// Shared post-response drain budget for a response whose body is already
    /// drained.  Same *order* as the thread path's post-response budgets, but not
    /// the same numbers: the thread path uses a 20 ms residual drain
    /// (`drain_residual`) and a 150 ms one-shot linger (`ONE_SHOT_LINGER`).
    /// These encode the RST-vs-FIN tradeoff, so the divergences are deliberate.
    const LINGER_BUDGET: Duration = Duration::from_millis(100);
    /// Explicit connection phase (replaces the former `u8` plus three ad-hoc
    /// constants).  The reactor's transitions are exactly
    /// `Reading → Writing → Draining → Reading` (keep-alive) or closed;
    /// a queued protocol error already has `keep_alive == false`, so it goes
    /// `Writing → Draining` and never back to `Reading`.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum ConnState {
        /// Accumulating a request header.
        Reading,
        /// Streaming a prepared response.
        Writing,
        /// Linger-draining unread inbound bytes before close.
        Draining,
    }

    /// The lazy deadline heap.  Entries carry `(deadline, fd, generation)`:
    /// `fd` alone is not an identity — a closed connection's descriptor can be
    /// reused by a new one whose `Instant` deadline happens to be equal (rare
    /// with a monotonic clock, but the ABA is free to rule out).  A popped entry
    /// is live only when the fd *and* the generation still match.
    type DeadlineHeap = BinaryHeap<Reverse<(Instant, i32, u64)>>;

    struct Conn {
        stream: TcpStream,
        /// Monotonic per-connection id; see `DeadlineHeap`.
        generation: u64,
        buf: Vec<u8>,
        /// Response kept alive for the write duration; slices are written
        /// directly with `write_vectored` (head, Connection header, body).
        response: Option<HttpResponse>,
        /// Date snapshot for the in-flight response; fixed for all partial
        /// writes so the `progress[3]` cursor always indexes one string.
        date: Option<std::sync::Arc<str>>,
        conn_hdr: &'static [u8],
        progress: [usize; 6],
        state: ConnState,
        keep_alive: bool,
        consumed: usize,
        deadline: Instant,
        /// Absolute deadline for the request header currently being read.
        /// `None` until the first byte of a request arrives, so an idle
        /// keep-alive connection gets a fresh budget per request.
        header_deadline: Option<Instant>,
        /// Watermark for incremental header-terminator scanning.
        header_scan: HeaderScan,
        /// Over-cap connection waiting for a first byte: silent preconnects are
        /// closed without a response, real requests get the cached 503.
        overload: bool,
        /// Unread request-body bytes (`usize::MAX` for ambiguous/chunked
        /// framing, where only EOF/budget can complete the drain).
        body_remaining: usize,
    }

    impl Conn {
        fn queue_bad_request(&mut self, fd: i32, heap: &mut DeadlineHeap) {
            self.response = Some(bad_request_response());
            self.conn_hdr = b"Connection: close\r\n\r\n";
            self.keep_alive = false;
            self.buf.clear();
            self.buf.shrink_to_fit();
            self.progress = [0; 6];
            self.consumed = 0;
            self.body_remaining = usize::MAX;
            self.state = ConnState::Writing;
            arm_deadline(&mut *self, fd, Instant::now() + WRITE_TIMEOUT, heap);
        }

        fn queue_too_large(&mut self, fd: i32, heap: &mut DeadlineHeap) {
            self.response = Some(HttpResponse::from_cache(
                "431 Request Header Fields Too Large",
                "",
                too_large_page(),
            ));
            self.conn_hdr = b"Connection: close\r\n\r\n";
            self.keep_alive = false;
            self.buf.clear();
            self.buf.shrink_to_fit();
            self.progress = [0; 6];
            self.consumed = 0;
            self.body_remaining = usize::MAX;
            self.state = ConnState::Writing;
            arm_deadline(&mut *self, fd, Instant::now() + WRITE_TIMEOUT, heap);
        }
    }

    /// Arm `conn`'s deadline and record it in the lazy heap, so
    /// `epoll_wait`'s timeout is the earliest LIVE deadline without an O(n)
    /// scan per wakeup.  Entries are discarded on peek when the connection is
    /// gone or its current deadline differs (the heap may hold stale arms);
    /// a deadline that SHORTENS is pushed immediately, so a wakeup is never
    /// late.
    fn arm_deadline(conn: &mut Conn, fd: i32, at: Instant, heap: &mut DeadlineHeap) {
        conn.deadline = at;
        heap.push(Reverse((at, fd, conn.generation)));
    }

    // Lazy-invalidation deadline heap (see `arm_deadline`).
    let mut deadline_heap: DeadlineHeap = BinaryHeap::new();
    // Source of `Conn::generation`.  One increment per accepted connection.
    let mut next_generation: u64 = 0;

    let epfd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
    if epfd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let _epfd_guard = unsafe { OwnedFd::from_raw_fd(epfd) };
    listener.set_nonblocking(true)?;
    let listener_fd = listener.as_raw_fd();
    let mut interest = libc::epoll_event {
        events: libc::EPOLLIN as u32,
        u64: listener_fd as u64,
    };
    epoll_ctl_checked(epfd, libc::EPOLL_CTL_ADD, listener_fd, &mut interest)?;

    let mut conns: FxHashMap<i32, Conn> = FxHashMap::default();
    // Number of resident connections that are not `overload` slots.  The accept
    // path used to rescan the whole map (`conns.values().filter(...).count()`)
    // for every accepted connection — an O(n) pass on the event-loop thread
    // inside the accept burst for a value that is trivial to maintain.
    let mut normal_conns: usize = 0;
    let mut events: Vec<libc::epoll_event> = (0..MAX_EVENTS)
        .map(|_| libc::epoll_event { events: 0, u64: 0 })
        .collect();
    let mut consecutive_accept_errors = 0usize;
    let mut accept_backoff_ms = 50u64;
    let first_deadline = Instant::now() + Duration::from_secs(10);
    let mut served_any = false;
    let mut last_served = Instant::now();

    loop {
        let now = Instant::now();
        if let Some(idle) = idle {
            if !served_any && now >= first_deadline && conns.is_empty() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "explain server: no client connected before the first-request deadline",
                ));
            }
            if served_any && conns.is_empty() && now.saturating_duration_since(last_served) >= idle
            {
                return Ok(());
            }
        }
        let mut wait = Duration::from_secs(5);
        // Drop stale arms (connection closed, or re-armed since) and take the
        // earliest live deadline from the heap instead of scanning every
        // connection on every wakeup.
        while let Some(&Reverse((at, fd, epoch))) = deadline_heap.peek() {
            match conns.get(&fd) {
                Some(conn) if at == conn.deadline && epoch == conn.generation => break,
                _ => {
                    deadline_heap.pop();
                }
            }
        }
        // The lazy heap can only accumulate: every re-arm (first byte, headers,
        // write, keep-alive idle, drain linger) pushes another entry, and a
        // stale entry survives until its deadline reaches the front.  Live
        // connections are bounded (normal + overload slots), so without a bound
        // the heap would track recent request throughput instead.  Rebuild from
        // the live set when it outgrows a small multiple of the connection count;
        // the O(n) rebuild is amortised against the >= O(n) stale pops it
        // removes, so it can never fire more than once per O(n) pushes.
        const HEAP_COMPACT_MIN: usize = 64;
        const HEAP_COMPACT_FACTOR: usize = 4;
        if deadline_heap.len() > HEAP_COMPACT_MIN + conns.len() * HEAP_COMPACT_FACTOR {
            deadline_heap = conns
                .iter()
                .map(|(&fd, conn)| Reverse((conn.deadline, fd, conn.generation)))
                .collect();
        }
        if let Some(&Reverse((at, _, _))) = deadline_heap.peek() {
            wait = wait.min(at.saturating_duration_since(now));
        }
        if let Some(idle) = idle {
            let until = if served_any {
                last_served + idle
            } else {
                first_deadline
            };
            wait = wait.min(until.saturating_duration_since(now));
        }
        // Ceiling: a positive sub-millisecond remainder truncated to 0 makes
        // `epoll_wait` poll instead of block, burning wakeups until the deadline.
        let timeout_ms = usize::try_from(wait.as_nanos().div_ceil(1_000_000))
            .unwrap_or(5_000)
            .clamp(1, 5_000) as i32;
        let ready =
            unsafe { libc::epoll_wait(epfd, events.as_mut_ptr(), MAX_EVENTS as i32, timeout_ms) };
        if ready < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }

        let mut close: smallvec::SmallVec<[i32; 64]> = smallvec::SmallVec::new();
        for event in &events[..ready as usize] {
            let fd = event.u64 as i32;
            if fd == listener_fd {
                if event.events & (libc::EPOLLERR | libc::EPOLLHUP) as u32 != 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::ConnectionAborted,
                        "explain server: listener error",
                    ));
                }
                for _ in 0..ACCEPT_BATCH {
                    match accept_nonblocking(listener) {
                        Ok(stream) => {
                            consecutive_accept_errors = 0;
                            accept_backoff_ms = 50;
                            swallow!(stream.set_nodelay(true));
                            let conn_fd = stream.as_raw_fd();
                            // One deadline value for both the connection and
                            // the heap arm, so the lazy-invalidation check
                            // (`at == conn.deadline`) stays exact.
                            let first_byte_deadline = Instant::now() + FIRST_BYTE_READ_TIMEOUT;
                            let normal = normal_conns;
                            if normal >= MAX_CONCURRENT_CONNECTIONS {
                                if conns.len() >= MAX_CONCURRENT_CONNECTIONS * 2 {
                                    // Bound overload slots too: a preconnect
                                    // flood cannot grow the map without limit.
                                    swallow!(stream.shutdown(Shutdown::Both));
                                } else {
                                    conns.insert(
                                        conn_fd,
                                        Conn {
                                            stream,
                                            generation: next_generation,
                                            buf: Vec::new(),
                                            response: None,
                                            date: None,
                                            conn_hdr: b"",
                                            progress: [0; 6],
                                            state: ConnState::Reading,
                                            keep_alive: false,
                                            consumed: 0,
                                            deadline: first_byte_deadline,
                                            header_deadline: None,
                                            header_scan: HeaderScan::new(),
                                            overload: true,
                                            body_remaining: 0,
                                        },
                                    );
                                }
                            } else {
                                conns.insert(
                                    conn_fd,
                                    Conn {
                                        stream,
                                        generation: next_generation,
                                        buf: Vec::new(),
                                        response: None,
                                        date: None,
                                        conn_hdr: b"",
                                        progress: [0; 6],
                                        state: ConnState::Reading,
                                        keep_alive: false,
                                        consumed: 0,
                                        deadline: first_byte_deadline,
                                        header_deadline: None,
                                        header_scan: HeaderScan::new(),
                                        overload: false,
                                        body_remaining: 0,
                                    },
                                );
                                normal_conns += 1;
                            }
                            // A closed (shed) accept leaves no map entry; the
                            // arm is then pruned as stale on the next peek.
                            next_generation = next_generation.wrapping_add(1);
                            // Only map-resident connections need an arm; a shed
                            // accept has nothing to expire (the entry would be
                            // pruned as stale on the next peek).
                            if let Some(c) = conns.get(&conn_fd) {
                                deadline_heap.push(Reverse((
                                    first_byte_deadline,
                                    conn_fd,
                                    c.generation,
                                )));
                            }
                            let mut conn_event = libc::epoll_event {
                                events: if conns
                                    .get(&conn_fd)
                                    .is_some_and(|c| c.state == ConnState::Writing)
                                {
                                    libc::EPOLLOUT as u32
                                } else {
                                    libc::EPOLLIN as u32
                                },
                                u64: conn_fd as u64,
                            };
                            if epoll_ctl_checked(
                                epfd,
                                libc::EPOLL_CTL_ADD,
                                conn_fd,
                                &mut conn_event,
                            )
                            .is_err()
                            {
                                if let Some(conn) = conns.remove(&conn_fd) {
                                    if !conn.overload {
                                        normal_conns = normal_conns.saturating_sub(1);
                                    }
                                    swallow!(conn.stream.shutdown(Shutdown::Both));
                                }
                            }
                        }
                        Err(ref error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                        Err(ref error) if error.kind() == std::io::ErrorKind::Interrupted => {
                            continue;
                        }
                        Err(error)
                            if matches!(
                                error.raw_os_error(),
                                Some(libc::EMFILE | libc::ENFILE | libc::ENOBUFS | libc::ENOMEM)
                            ) =>
                        {
                            // Match the thread path: fd/memory exhaustion is
                            // environmental.  Back off and let success reset
                            // it; only a genuinely dead listener reaches the
                            // terminal counter.
                            std::thread::sleep(Duration::from_millis(accept_backoff_ms));
                            accept_backoff_ms = (accept_backoff_ms * 2).min(1000);
                            break;
                        }
                        Err(ref error)
                            if matches!(
                                error.raw_os_error(),
                                Some(libc::ECONNABORTED | libc::EPROTO)
                            ) =>
                        {
                            std::thread::sleep(Duration::from_millis(10));
                            break;
                        }
                        Err(error) => {
                            consecutive_accept_errors += 1;
                            if consecutive_accept_errors >= MAX_CONSECUTIVE_ACCEPT_ERRORS {
                                return Err(error);
                            }
                            std::thread::sleep(Duration::from_millis(10));
                            break;
                        }
                    }
                }
                continue;
            }

            let Some(conn) = conns.get_mut(&fd) else {
                continue;
            };
            let bits = event.events as i32;
            if (conn.state == ConnState::Reading || conn.state == ConnState::Draining)
                && bits & (libc::EPOLLERR | libc::EPOLLHUP) != 0
                && bits & libc::EPOLLIN == 0
            {
                close.push(fd);
                continue;
            }

            if conn.state == ConnState::Draining {
                // The response is fully written; consume whatever the client
                // still sends so the final close sends FIN instead of RST.
                let mut eof = false;
                loop {
                    let mut chunk = [0u8; 4096];
                    if Instant::now() >= conn.deadline {
                        // Leave the connection for the deadline sweep; a
                        // peer that keeps sending must not pin the reactor.
                        break;
                    }
                    match conn.stream.read(&mut chunk) {
                        Ok(0) => {
                            eof = true;
                            break;
                        }
                        Ok(read) => {
                            if conn.body_remaining != usize::MAX {
                                conn.body_remaining = conn.body_remaining.saturating_sub(read);
                                if conn.body_remaining == 0 {
                                    // Body consumed is not EOF: a pipelined
                                    // request or late bytes may still arrive.
                                    conn.body_remaining = usize::MAX;
                                }
                            }
                        }
                        Err(ref error) if error.kind() == std::io::ErrorKind::Interrupted => {
                            continue;
                        }
                        Err(ref error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                        Err(_) => {
                            eof = true;
                            break;
                        }
                    }
                }
                if eof {
                    close.push(fd);
                }
                continue;
            }

            if conn.state == ConnState::Reading {
                let mut eof = false;
                let mut header_progress = HeaderProgress::Incomplete;
                loop {
                    let mut chunk = [0u8; 8192];
                    let cap = MAX_HEADER + 8192;
                    let want = cap.saturating_sub(conn.buf.len()).min(chunk.len());
                    if want == 0 {
                        header_progress = HeaderProgress::TooLargeHeader;
                        break;
                    }
                    match conn.stream.read(&mut chunk[..want]) {
                        Ok(0) => {
                            eof = true;
                            break;
                        }
                        Ok(read) => conn.buf.extend_from_slice(&chunk[..read]),
                        Err(ref error) if error.kind() == std::io::ErrorKind::Interrupted => {
                            continue;
                        }
                        Err(ref error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                        Err(_) => {
                            eof = true;
                            break;
                        }
                    }
                    header_progress = conn.header_scan.scan(&conn.buf);
                    match header_progress {
                        HeaderProgress::Complete(_)
                        | HeaderProgress::TooLargeHeader
                        | HeaderProgress::TooLargeLine => break,
                        HeaderProgress::Incomplete => {}
                    }
                }
                if matches!(
                    header_progress,
                    HeaderProgress::TooLargeHeader | HeaderProgress::TooLargeLine
                ) {
                    conn.queue_too_large(fd, &mut deadline_heap);
                    let mut event = libc::epoll_event {
                        events: libc::EPOLLOUT as u32,
                        u64: fd as u64,
                    };
                    if epoll_ctl_checked(epfd, libc::EPOLL_CTL_MOD, fd, &mut event).is_err() {
                        close.push(fd);
                    }
                    continue;
                }
                if eof && conn.buf.is_empty() {
                    close.push(fd);
                    continue;
                }
                let Some(head_end) = (match header_progress {
                    HeaderProgress::Complete(end) => Some(end),
                    HeaderProgress::Incomplete
                    | HeaderProgress::TooLargeHeader
                    | HeaderProgress::TooLargeLine => None,
                }) else {
                    if eof {
                        close.push(fd);
                    } else if !conn.buf.is_empty() {
                        // First byte seen: the header budget starts now and is
                        // never extended by later reads.  Push the heap entry
                        // only on that transition: a client dribbling its
                        // header one byte at a time produces one readiness event
                        // per byte, and re-pushing the *same* deadline each time
                        // grew the heap with the number of reads instead of the
                        // number of connections (the lazy stale sweep stops at
                        // an entry whose deadline still matches the live one, so
                        // it cannot discard the duplicates).
                        let first_byte = conn.header_deadline.is_none();
                        let header_deadline = *conn
                            .header_deadline
                            .get_or_insert_with(|| Instant::now() + Duration::from_secs(2));
                        conn.deadline = header_deadline;
                        if first_byte {
                            deadline_heap.push(Reverse((header_deadline, fd, conn.generation)));
                        }
                    }
                    continue;
                };
                if conn.overload {
                    let is_head = request_line_is_head(&conn.buf);
                    let framing = scan_framing(&conn.buf[..head_end], true);
                    let already_read = conn.buf.len().saturating_sub(head_end);
                    conn.body_remaining = match framing.body(already_read) {
                        RequestBody::Unframed => usize::MAX,
                        RequestBody::Empty => 0,
                        RequestBody::Fixed { remaining } => remaining,
                    };
                    conn.buf.clear();
                    conn.buf.shrink_to_fit();
                    let page = shed_page();
                    let response = HttpResponse::from_cache("503 Service Unavailable", "", page);
                    conn.response = Some(if is_head {
                        response.as_head()
                    } else {
                        response
                    });
                    conn.conn_hdr = b"Connection: close\r\n\r\n";
                    conn.keep_alive = false;
                    // No `conn.consumed` here: the buffer was cleared above, and the field
                    // is only read by the keep-alive reuse assert — which this path can
                    // never reach.  Setting it anyway pointed past the end of the buffer.
                    conn.consumed = 0;
                    conn.progress = [0; 6];
                    conn.state = ConnState::Writing;
                    conn.header_deadline = None;
                    arm_deadline(conn, fd, Instant::now() + WRITE_TIMEOUT, &mut deadline_heap);
                    let mut conn_event = libc::epoll_event {
                        events: libc::EPOLLOUT as u32,
                        u64: fd as u64,
                    };
                    if epoll_ctl_checked(epfd, libc::EPOLL_CTL_MOD, fd, &mut conn_event).is_err() {
                        close.push(fd);
                    }
                    continue;
                }
                let (response, framing, http11) = {
                    let head = &conn.buf[..head_end];
                    let line_end = memchr::memchr(b'\n', head)
                        .map(|position| {
                            if position > 0 && head[position - 1] == b'\r' {
                                position - 1
                            } else {
                                position
                            }
                        })
                        .unwrap_or(0);
                    if line_end >= MAX_REQUEST_LINE {
                        // Parity with the other backends: 431 instead of a
                        // bare close, then drain under the shared linger
                        // budget so the response is not RST away.
                        conn.queue_too_large(fd, &mut deadline_heap);
                        let mut event = libc::epoll_event {
                            events: libc::EPOLLOUT as u32,
                            u64: fd as u64,
                        };
                        if epoll_ctl_checked(epfd, libc::EPOLL_CTL_MOD, fd, &mut event).is_err() {
                            close.push(fd);
                        }
                        continue;
                    }
                    let Ok(request_line) = simdutf8::basic::from_utf8(&head[..line_end]) else {
                        conn.queue_bad_request(fd, &mut deadline_heap);
                        let mut event = libc::epoll_event {
                            events: libc::EPOLLOUT as u32,
                            u64: fd as u64,
                        };
                        if epoll_ctl_checked(epfd, libc::EPOLL_CTL_MOD, fd, &mut event).is_err() {
                            close.push(fd);
                        }
                        continue;
                    };
                    let (response, http11) = dispatch_request_meta(request_line);
                    (
                        response,
                        // One framing policy for all backends: the thread
                        // path's `scan_framing` decides body/TE/Connection
                        // semantics here too.
                        // Slice starts *after* the request line: the first line
                        // here is a header, so a colon-less first line must be
                        // flagged malformed instead of skipped as a request line.
                        scan_framing(&head[line_end..head_end], false),
                        http11,
                    )
                };
                let already_read = conn.buf.len().saturating_sub(head_end);
                conn.body_remaining = if matches!(framing.body(already_read), RequestBody::Unframed)
                {
                    usize::MAX
                } else {
                    framing
                        .content_length
                        .unwrap_or(0)
                        .saturating_sub(already_read)
                };
                let decision = framing.decide(already_read, http11);
                let keep = response_allows_keep_alive(response.status)
                    && decision.keep_alive_allowed
                    && conn.buf.len() == head_end;
                conn.keep_alive = keep;
                if keep {
                    // The field is the whole point of the keep-alive reuse path: reuse
                    // requires the buffer to end exactly at the header terminator, so
                    // `drain(..consumed)` shifts nothing.  Assert it here, where both
                    // quantities are live, instead of only at the distant reuse site.
                    debug_assert_eq!(
                        conn.buf.len(),
                        head_end,
                        "keep-alive reuse with a pipelined remainder would lose bytes"
                    );
                    conn.consumed = conn.buf.len();
                } else {
                    conn.buf.clear();
                    conn.buf.shrink_to_fit();
                    conn.consumed = 0;
                }
                conn.response = Some(response);
                conn.conn_hdr = if keep {
                    b"Connection: keep-alive\r\n\r\n"
                } else {
                    b"Connection: close\r\n\r\n"
                };
                conn.progress = [0; 6];
                conn.state = ConnState::Writing;
                arm_deadline(conn, fd, Instant::now() + WRITE_TIMEOUT, &mut deadline_heap);
                let mut conn_event = libc::epoll_event {
                    events: libc::EPOLLOUT as u32,
                    u64: fd as u64,
                };
                if epoll_ctl_checked(epfd, libc::EPOLL_CTL_MOD, fd, &mut conn_event).is_err() {
                    close.push(fd);
                }
            }

            if conn.state == ConnState::Writing {
                let mut finished = false;
                let date = match conn.date.as_ref() {
                    Some(date) => std::sync::Arc::clone(date),
                    None => {
                        let date = http_date_header();
                        conn.date = Some(std::sync::Arc::clone(&date));
                        date
                    }
                };
                match conn.response.as_ref() {
                    None => finished = true,
                    Some(response) => {
                        let head = response.head_parts();
                        let body = response.body().as_bytes();
                        let lengths = [
                            head[0].len(),
                            head[1].len(),
                            head[2].len(),
                            date.len(),
                            conn.conn_hdr.len(),
                            body.len(),
                        ];
                        loop {
                            if conn.progress[0] >= lengths[0]
                                && conn.progress[1] >= lengths[1]
                                && conn.progress[2] >= lengths[2]
                                && conn.progress[3] >= lengths[3]
                                && conn.progress[4] >= lengths[4]
                                && conn.progress[5] >= lengths[5]
                            {
                                finished = true;
                                break;
                            }
                            let slices = [
                                std::io::IoSlice::new(&head[0].as_bytes()[conn.progress[0]..]),
                                std::io::IoSlice::new(&head[1].as_bytes()[conn.progress[1]..]),
                                std::io::IoSlice::new(&head[2].as_bytes()[conn.progress[2]..]),
                                std::io::IoSlice::new(&date.as_bytes()[conn.progress[3]..]),
                                std::io::IoSlice::new(&conn.conn_hdr[conn.progress[4]..]),
                                std::io::IoSlice::new(&body[conn.progress[5]..]),
                            ];
                            match conn.stream.write_vectored(&slices) {
                                Ok(0) => {
                                    close.push(fd);
                                    break;
                                }
                                Ok(mut written) => {
                                    for (i, len) in lengths.iter().enumerate() {
                                        let take = (len - conn.progress[i]).min(written);
                                        conn.progress[i] += take;
                                        written -= take;
                                        if written == 0 {
                                            break;
                                        }
                                    }
                                }
                                Err(ref error)
                                    if error.kind() == std::io::ErrorKind::Interrupted =>
                                {
                                    continue;
                                }
                                Err(ref error)
                                    if error.kind() == std::io::ErrorKind::WouldBlock =>
                                {
                                    break;
                                }
                                Err(_) => {
                                    close.push(fd);
                                    break;
                                }
                            }
                        }
                    }
                }
                if finished {
                    served_any = true;
                    last_served = Instant::now();
                    if conn.keep_alive {
                        // Keep-alive reuse requires the buffer to end exactly at
                        // the header terminator (no pipelined remainder), so
                        // `consumed` covers the whole buffer and
                        // `drain(..consumed)` only shifted nothing.  Assert the
                        // invariant and clear instead: `clear` keeps the
                        // allocation without a memmove.
                        debug_assert_eq!(
                            conn.consumed,
                            conn.buf.len(),
                            "keep-alive reuse with a pipelined remainder would lose bytes"
                        );
                        conn.buf.clear();
                        conn.header_scan.reset();
                        conn.response = None;
                        conn.date = None;
                        conn.progress = [0; 6];
                        conn.state = ConnState::Reading;
                        conn.header_deadline = None;
                        arm_deadline(
                            conn,
                            fd,
                            Instant::now() + KEEP_ALIVE_IDLE,
                            &mut deadline_heap,
                        );
                        let mut conn_event = libc::epoll_event {
                            events: libc::EPOLLIN as u32,
                            u64: fd as u64,
                        };
                        if epoll_ctl_checked(epfd, libc::EPOLL_CTL_MOD, fd, &mut conn_event)
                            .is_err()
                        {
                            close.push(fd);
                        }
                    } else {
                        // Half-close and drain inbound bytes: closing with
                        // unread request/body bytes makes the kernel emit RST,
                        // which can discard the response we just wrote (the
                        // shed 503 path included).
                        swallow!(conn.stream.shutdown(Shutdown::Write));
                        conn.state = ConnState::Draining;
                        let linger = if conn.body_remaining == 0 {
                            LINGER_BUDGET
                        } else {
                            Duration::from_secs(1)
                        };
                        arm_deadline(conn, fd, Instant::now() + linger, &mut deadline_heap);
                        let mut conn_event = libc::epoll_event {
                            events: libc::EPOLLIN as u32,
                            u64: fd as u64,
                        };
                        if epoll_ctl_checked(epfd, libc::EPOLL_CTL_MOD, fd, &mut conn_event)
                            .is_err()
                        {
                            close.push(fd);
                        }
                    }
                }
            }
        }

        // Expiry is driven by the deadline heap, not by a scan of every live
        // connection: pop entries whose deadline has passed and close the
        // connection only when the entry still describes its *current*
        // deadline.  Stale entries (a connection that re-armed since) are
        // dropped without touching the map, so a wakeup costs
        // O(expired + stale) instead of O(connections).
        //
        // Sound only because every write to `conn.deadline` also pushes a heap
        // entry (`Conn::arm_deadline`, plus the first-byte header deadline), so
        // a live deadline is always represented by at least one entry.
        let now = Instant::now();
        while let Some(&Reverse((deadline, fd, epoch))) = deadline_heap.peek() {
            if deadline > now {
                break;
            }
            deadline_heap.pop();
            if conns
                .get(&fd)
                .is_some_and(|conn| conn.deadline == deadline && conn.generation == epoch)
            {
                close.push(fd);
            }
        }
        close.sort_unstable();
        close.dedup();
        for fd in close {
            if let Some(conn) = conns.remove(&fd) {
                if !conn.overload {
                    normal_conns = normal_conns.saturating_sub(1);
                }
                swallow!(epoll_ctl_checked(epfd, libc::EPOLL_CTL_DEL, fd, std::ptr::null_mut()));
                swallow!(conn.stream.shutdown(Shutdown::Both));
            }
        }
    }
}

fn handle_persistent_connection(
    stream: &mut TcpStream,
    served: &AtomicUsize,
) -> std::io::Result<()> {
    // Accepted sockets inherit O_NONBLOCK on some BSD/macOS listeners; the
    // blocking handler plus read/write timeouts is the intended regime.
    stream.set_nonblocking(false)?;
    swallow!(stream.set_nodelay(true));
    // A client that stops reading must not wedge `write_all` on a full
    // socket buffer; the response is a few KB, well under one write, but a
    // pathological client can still refuse to drain it.
    stream.set_write_timeout(Some(Duration::from_secs(1)))?;
    // : HTTP/1.1 keep-alive.  A browser loading the index plus five code
    // pages used to open six TCP connections; reuse costs one handshake.
    // The idle slot is released after `KEEP_ALIVE_IDLE` so an idle socket
    // cannot starve `MAX_CONCURRENT_CONNECTIONS`.
    let mut first_request = true;
    loop {
        // The first request may follow a speculative preconnect; subsequent
        // requests on a reused connection are normally already queued, so the
        // longer timeout only matters when the client goes idle.
        let budget = if first_request {
            FIRST_BYTE_READ_TIMEOUT
        } else {
            KEEP_ALIVE_IDLE
        };
        stream.set_read_timeout(Some(budget))?;
        first_request = false;
        let mut too_large = false;
        let mut malformed = false;
        let Some((line, leftover)) =
            read_request_line(stream, &[], budget, &mut too_large, &mut malformed)
        else {
            if too_large {
                swallow!(write_header_too_large(stream));
                linger_after_half_close(stream);
            } else if malformed {
                swallow!(write_prepared(stream, bad_request_head(), ""));
                linger_after_half_close(stream);
            }
            return Ok(());
        };
        // Drain the rest of this request — header block *and* any body
        // announced by `Content-Length` — before responding.  Closing a socket
        // with unread inbound data makes the kernel send RST instead of FIN.
        // `leftover` is what `read_request_line` pulled past the request `\n`.
        let mut tail_len = 0usize;
        let mut wants_close = false;
        let mut is_head = false;
        let http11 = line.trim_ascii_end().ends_with("HTTP/1.1");
        let outcome = drain_request_tracked(
            stream,
            &leftover,
            Duration::from_secs(1),
            &mut tail_len,
            &mut wants_close,
            &mut is_head,
            false,
            Some(http11),
            // `MAX_HEADER` covers the whole head; this path already consumed the
            // request line, so its length has to count too.
            line.len(),
        );
        if outcome == DrainOutcome::NoRequest {
            return Ok(());
        }
        if outcome == DrainOutcome::TooLarge {
            swallow!(write_header_too_large(stream));
            linger_after_half_close(stream);
            return Ok(());
        }
        let (response, http11) = dispatch_request_meta(&line);
        let keep_alive = outcome == DrainOutcome::Complete
            && response_allows_keep_alive(response.status)
            && http11
            && tail_len == 0
            && !wants_close;
        // : write the pre-rendered head + body — no per-request formatting.
        write_prepared_conn(stream, response.head_parts(), response.body(), keep_alive)?;
        served.fetch_add(1, Ordering::Relaxed);
        if keep_alive {
            // Do NOT `drain_residual`: bytes arriving after this response
            // belong to the next request on the same connection (browsers do
            // not pipeline, so nothing is expected, but a 20 ms read could
            // swallow an early next request).
            continue;
        }
        finish_after_response(stream, outcome, false)?;
        return Ok(());
    }
}

/// A syntactically complete HTTP/1.x request line (exactly three tokens).
struct ParsedRequestLine<'a> {
    method: &'a str,
    target: &'a str,
    version: &'a str,
}

/// One-pass request-line grammar: exactly `method target version`, no extras.
///
/// Deliberately uses ASCII-whitespace splitting rather than a single-SP
/// grammar: the loopback one-shot and persistent backends share this parser
/// and must accept the same bytes. Tightening one backend alone would create
/// a cross-platform routing difference for `GET\t/E030`.
/// Whether a raw request line names the `HEAD` method.
///
/// The separator may be SP **or HTAB**, matching [`parse_request_line`].  The
/// two `starts_with(b"HEAD ")` checks this replaces accepted only a space, so a
/// tab-separated HEAD was classified as a normal request and received a
/// body-bearing response (including a body-bearing 503 on the overload path).
fn request_line_is_head(line: &[u8]) -> bool {
    matches!(
        line.strip_prefix(b"HEAD").and_then(|rest| rest.first()),
        Some(b' ' | b'\t')
    )
}

/// Whether `s` is an RFC 7230 `token` (a method must be one).
fn is_http_token(s: &str) -> bool {
    !s.is_empty()
        && s.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(
                    b,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}

fn parse_request_line(line: &str) -> Option<ParsedRequestLine<'_>> {
    // SP/HTAB only.  `split_ascii_whitespace` also treats VT (0x0B) and FF
    // (0x0C) as separators, which the one-shot prefilter rejects — accepting a
    // request line there but not here is how the two backends drift apart.
    let mut parts = line.split([' ', '\t']).filter(|part| !part.is_empty());
    let method = parts.next()?;
    let target = parts.next()?;
    let version = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    if !is_http_token(method) {
        // An invalid method is a malformed request line (400), not a valid but
        // unsupported one (405).
        return None;
    }
    Some(ParsedRequestLine {
        method,
        target,
        version,
    })
}

/// Pre-rendered 400 head (shared security headers) for malformed request
/// lines in the routed one-shot path.
fn bad_request_head() -> &'static str {
    static HEAD: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    HEAD.get_or_init(|| render_head("400 Bad Request", "", 0))
}

/// Whether the buffered header prefix contains `Connection: close`.
///
/// The header block is not retained (the drain discards it), but the bytes
/// after the request line that arrived with it are available; a browser that
/// asks to close is then honored instead of being kept alive.
/// Read one line (up to and including the `\r\n`) from the stream, plus any
/// bytes the same read pulled past the `\n`.
///
/// Returns `(line, leftover)`: `line` is the request line without the
/// trailing CRLF, and `leftover` is everything after the `\n` that arrived
/// in the same `read` chunk.  Those bytes belong to the request headers and
/// must be handed to [`drain_request`] as its `prefix` — otherwise the next
/// `read` would skip them and the drain could miss the `\r\n\r\n` terminator.
///
/// Reads in 1 KiB chunks (one `read` syscall per chunk) instead of one
/// byte at a time, so a typical ~20–80 byte request line is one syscall and a
/// long query string needs half as many reads.  `line` is a
/// `SmallVec<[u8; 512]>` so it stays on the stack for the common request-line
/// length; `leftover` is bounded by the chunk size and spills only when a
/// single read crosses the inline capacity.
fn read_request_line(
    stream: &mut TcpStream,
    prefix: &[u8],
    budget: Duration,
    too_large: &mut bool,
    malformed: &mut bool,
) -> Option<(CompactString, SmallVec<[u8; 512]>)> {
    // A caller may already hold the line prefix from an earlier probe
    // (the 512-byte one-shot read).  Handle a line completed inside it without
    // another syscall, and seed the accumulator otherwise.
    if let Some(nl) = memchr::memchr(b'\n', prefix) {
        if nl >= MAX_REQUEST_LINE {
            *too_large = true;
            swallow!(drain_after_request_line(stream, &prefix[nl + 1..], Duration::from_millis(100)));
            return None;
        }
        let raw = &prefix[..nl];
        let raw = raw.strip_suffix(b"\r").unwrap_or(raw);
        let Ok(line) = simdutf8::basic::from_utf8(raw) else {
            *malformed = true;
            swallow!(drain_after_request_line(stream, &prefix[nl + 1..], Duration::from_millis(100)));
            return None;
        };
        let mut leftover: SmallVec<[u8; 512]> = SmallVec::new();
        leftover.extend_from_slice(&prefix[nl + 1..]);
        return Some((line.into(), leftover));
    }
    // Bound the *total* time spent reading a request line, not just each
    // `read`: a client trickling one byte every ~499 ms could otherwise hold a
    // connection for minutes.  Combined with the inline fallback at
    // `MAX_CONCURRENT_CONNECTIONS`, that was a local DoS on the accept loop.
    // The caller chooses the budget: first-request preconnect detection uses
    // `FIRST_BYTE_READ_TIMEOUT`, keep-alive uses `KEEP_ALIVE_IDLE`.
    let deadline = Instant::now() + budget;
    let mut line: SmallVec<[u8; 512]> = SmallVec::new();
    line.extend_from_slice(prefix);
    let mut buf = [0u8; 1024];
    loop {
        if Instant::now() >= deadline {
            // A partial request line may still have bytes queued in the
            // kernel.  Drain them before closing so the peer sees a FIN, not
            // an RST.
            if !line.is_empty() {
                swallow!(drain_request(stream, &[], Duration::from_millis(100)));
            }
            return None;
        }
        let n = {
            if arm_read_timeout_until(stream, deadline).is_err() {
                if !line.is_empty() {
                    swallow!(drain_request(stream, &[], Duration::from_millis(100)));
                }
                return None;
            }
            match stream.read(&mut buf) {
                Ok(0) => return None,
                Ok(n) => n,
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => {
                    // drain any partial line before closing, matching the
                    // deadline path; otherwise bytes that raced into the kernel
                    // buffer are discarded with unread inbound data → RST.
                    if !line.is_empty() {
                        swallow!(drain_request(stream, &[], Duration::from_millis(100)));
                    }
                    return None;
                }
            }
        };
        let chunk = &buf[..n];
        match memchr::memchr(b'\n', chunk) {
            Some(nl) => {
                if line.len() + nl >= MAX_REQUEST_LINE {
                    *too_large = true;
                    // The request line is over the cap.  `chunk[nl + 1..]`
                    // arrived past the newline and the kernel may hold more;
                    // drain both before returning so closing does not reset
                    // the connection under the client.
                    swallow!(drain_after_request_line(
                        stream,
                        &chunk[nl + 1..],
                        Duration::from_millis(100),
                    ));
                    return None;
                }
                line.extend_from_slice(&chunk[..nl]);
                if line.ends_with(b"\r") {
                    line.pop();
                }
                let mut leftover: SmallVec<[u8; 512]> = SmallVec::new();
                leftover.extend_from_slice(&chunk[nl + 1..]);
                // Invalid UTF-8 is a malformed request, not a reason to leave
                // the rest of the request unread (which would RST on close).
                let Ok(line) = simdutf8::basic::from_utf8(&line[..]) else {
                    *malformed = true;
                    swallow!(drain_after_request_line(stream, &leftover, Duration::from_millis(100)));
                    return None;
                };
                return Some((line.into(), leftover));
            }
            None => {
                if line.len() + n >= MAX_REQUEST_LINE {
                    *too_large = true;
                    swallow!(drain_request(stream, &[], Duration::from_millis(100)));
                    return None;
                }
                line.extend_from_slice(chunk);
            }
        }
    }
}

/// Owned head split into the three vectored-write parts.  The HTTP head is a
/// single allocation; the other two slices are reserved for the writer's
/// stable `[head, connection, body]` shape and are always empty.
type HeadParts = [std::sync::Arc<str>; 3];

fn head_parts_owned(head: String) -> HeadParts {
    [
        std::sync::Arc::from(head),
        std::sync::Arc::from(""),
        std::sync::Arc::from(""),
    ]
}

/// Owned page for the bounded 404 cache.
///
/// The explain/index pages are process-immortal and can stay `Box::leak`ed,
/// but 404 entries are *evicted* by `NOT_FOUND_CACHE_CAP` — leaking them there
/// made the cap meaningless and leaked one page per distinct path forever.
/// Owning `Arc<str>` lets eviction actually reclaim the strings.
struct OwnedPage {
    head: HeadParts,
    body: std::sync::Arc<str>,
}

impl OwnedPage {
    fn head_parts(&self) -> [&str; 3] {
        [&self.head[0], &self.head[1], &self.head[2]]
    }
}

/// A fully rendered response for a cached page: the head already
/// contains the correct `Content-Length`, so a hot request is two pointer
/// copies and one vectored write — no lock, no hash, no formatting.
///
/// The page owns its strings (`Box<str>`) rather than `Box::leak`ing all
/// three allocations.  The surrounding `OnceLock` is process-long anyway, so
/// residency is unchanged while ownership/destruction become explicit.
struct CachedPage {
    head: HeadParts,
    body: Box<str>,
}

impl CachedPage {
    fn head_parts(&self) -> [&str; 3] {
        [&self.head[0], &self.head[1], &self.head[2]]
    }

    fn build(status: &str, extra_headers: &str, body: String) -> Self {
        let head = render_head(status, extra_headers, body.len());
        CachedPage {
            head: head_parts_owned(head),
            body: body.into_boxed_str(),
        }
    }
}

/// Serve a cached explain page for `code`.
///
/// `build_explain_page` is a pure function of the static error-code tables,
/// so each code's page is built at most once and reused for the lifetime of
/// the persistent server.
///
/// : the valid code set is static and small, and `ErrCode::try_new` resolves
/// it through the registry's direct-address index — so the cache is a
/// slot-indexed array of `OnceLock`s, not a `Mutex<HashMap>`. The hot path is
/// one array index + one atomic load + one `Arc` refcount bump: no lock, no
/// hash, no key allocation. A per-slot `OnceLock` also serializes concurrent
/// first builds of the same code.
fn explain_page_cached(code: &str) -> Option<&'static CachedPage> {
    use crate::diagnostics::error_code::{DIRECT_ADDRESS_SLOTS, direct_slot};
    let ec = ErrCode::try_new(code).ok()?;
    let slot = direct_slot(ec.code())?;
    type PageSlot = std::sync::OnceLock<CachedPage>;
    static PAGES: std::sync::OnceLock<Box<[PageSlot]>> = std::sync::OnceLock::new();
    let pages = PAGES.get_or_init(|| {
        (0..DIRECT_ADDRESS_SLOTS)
            .map(|_| std::sync::OnceLock::new())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    });
    let cell = &pages[slot];
    Some(cell.get_or_init(|| CachedPage::build("200 OK", "", build_explain_page(&ec))))
}

/// A routed HTTP response. The page is referenced once: a static explain
/// page borrows its immortal `CachedPage`, a cached 404 shares one
/// `Arc<OwnedPage>` — no second `Arc` clone per response.
enum PageRef {
    Static(&'static CachedPage),
    Shared(std::sync::Arc<OwnedPage>),
}

impl PageRef {
    fn head_parts(&self) -> [&str; 3] {
        let head = match self {
            PageRef::Static(page) => &page.head,
            PageRef::Shared(page) => &page.head,
        };
        [&head[0], &head[1], &head[2]]
    }

    fn body(&self) -> &str {
        match self {
            PageRef::Static(page) => page.body.as_ref(),
            PageRef::Shared(page) => page.body.as_ref(),
        }
    }
}

struct HttpResponse {
    status: &'static str,
    extra_headers: &'static str,
    page: PageRef,
    /// HEAD must echo the GET head (including Content-Length) with no body.
    head_only: bool,
}

impl HttpResponse {
    fn from_cache(
        status: &'static str,
        extra_headers: &'static str,
        page: &'static CachedPage,
    ) -> Self {
        // `extra_headers` records *why* this head carries the headers it does, but it was
        // write-only (one test read it), so the field and the prebuilt head could drift
        // apart without a symptom.  `page.head` is rendered from the same argument, so the
        // linkage is checkable at construction for free.
        debug_assert!(
            extra_headers.is_empty() || page.head.iter().any(|part| part.contains(extra_headers)),
            "the prebuilt head does not carry the stored extra_headers"
        );
        HttpResponse {
            status,
            extra_headers,
            page: PageRef::Static(page),
            head_only: false,
        }
    }

    /// Share the cached page's single allocation instead of cloning
    /// both `Arc` fields.
    fn from_owned(
        status: &'static str,
        extra_headers: &'static str,
        page: std::sync::Arc<OwnedPage>,
    ) -> Self {
        HttpResponse {
            status,
            extra_headers,
            page: PageRef::Shared(page),
            head_only: false,
        }
    }

    fn as_head(mut self) -> Self {
        self.head_only = true;
        self
    }

    fn head_parts(&self) -> [&str; 3] {
        self.page.head_parts()
    }

    fn body(&self) -> &str {
        if self.head_only { "" } else { self.page.body() }
    }
}

fn bad_request_response() -> HttpResponse {
    static PAGE: std::sync::OnceLock<CachedPage> = std::sync::OnceLock::new();
    HttpResponse::from_cache(
        "400 Bad Request",
        "",
        PAGE.get_or_init(|| {
            CachedPage::build(
                "400 Bad Request",
                "",
                simple_page(
                    "400 — bad request",
                    "400 — bad request",
                    "The request line is malformed. Use <code>GET /&lt;code&gt; HTTP/1.1</code>.",
                ),
            )
        }),
    )
}

fn method_not_allowed_response() -> HttpResponse {
    static PAGE: std::sync::OnceLock<CachedPage> = std::sync::OnceLock::new();
    HttpResponse::from_cache(
        "405 Method Not Allowed",
        "Allow: GET, HEAD\r\n",
        PAGE.get_or_init(|| {
            CachedPage::build(
                "405 Method Not Allowed",
                "Allow: GET, HEAD\r\n",
                simple_page(
                    "405 — method not allowed",
                    "405 — method not allowed",
                    "This server only accepts <code>GET</code> and <code>HEAD</code> requests.",
                ),
            )
        }),
    )
}

/// Route a request line to its page.
///
/// The request line must have exactly three whitespace-separated fields
/// (`method SP request-target SP HTTP-version`); anything else is a `400`
/// rather than being silently routed to the index as the old
/// `nth(1).unwrap_or("/")` did.  `GET` and `HEAD` are served; other methods
/// get a `405` with an `Allow` header.
#[cfg(test)]
fn dispatch_request(request_line: &str) -> HttpResponse {
    dispatch_request_meta(request_line).0
}

/// Statuses that may keep the connection alive.  Shared by the blocking
/// thread backend and the epoll reactor so one cannot silently diverge on
/// which protocol-error responses are reusable.
fn response_allows_keep_alive(status: &str) -> bool {
    matches!(
        status,
        "200 OK" | "400 Bad Request" | "404 Not Found" | "405 Method Not Allowed"
    )
}

fn dispatch_request_meta(request_line: &str) -> (HttpResponse, bool) {
    // One shared three-token parser for both servers.
    //
    // This arm **is** reachable: `parse_request_line` returns `None` for a malformed
    // three-token line (bad method token, missing target, junk version), and the caller
    // reaches here with the raw line after the UTF-8 check.  It is not a defensive `else`
    // for an "impossible" case — a review of this function read it as unreachable; the
    // `invalid_utf8_and_syntax_errors_share_one_bad_request_shape` test exercises the
    // sibling early exit, and this one answers the same 400 shape.
    let Some(req) = parse_request_line(request_line) else {
        return (bad_request_response(), false);
    };
    let http11 = req.version == "HTTP/1.1";
    let head_only = req.method == "HEAD";
    if !req.target.starts_with('/') || !matches!(req.version, "HTTP/1.0" | "HTTP/1.1") {
        let response = bad_request_response();
        return (
            if head_only {
                response.as_head()
            } else {
                response
            },
            http11,
        );
    }
    if req.method != "GET" && !head_only {
        return (method_not_allowed_response(), http11);
    }
    let target = req.target;

    // `GET /E030?view=full HTTP/1.1` → canonical code `E030`.  The shared
    // helper applies the same query/percent/slash policy as the one-shot
    // router ; `//` names a different resource and must 404, not index.
    let raw = target.split('?').next().unwrap_or("/");
    if raw == "/" {
        let response = HttpResponse::from_cache("200 OK", "", build_index_page());
        return (
            if head_only {
                response.as_head()
            } else {
                response
            },
            http11,
        );
    }
    let response = match canonical_path_code(target) {
        Some(code) => match explain_page_cached(code) {
            Some(page) => HttpResponse::from_cache("200 OK", "", page),
            None => HttpResponse::from_owned("404 Not Found", "", not_found_page_cached(code)),
        },
        None => {
            let decoded = percent_decode(raw);
            let key = decoded
                .as_str()
                .strip_prefix('/')
                .unwrap_or(decoded.as_str());
            let key = if key.is_empty() { "/" } else { key };
            HttpResponse::from_owned("404 Not Found", "", not_found_page_cached(key))
        }
    };
    (
        if head_only {
            response.as_head()
        } else {
            response
        },
        http11,
    )
}

/// Decode one ASCII hex digit.  Unlike `u8::from_str_radix`, this rejects a
/// leading `+` (which `from_str_radix` accepts, so `%+F` used to decode to
/// `0x0F` instead of remaining literal invalid input).
#[inline]
fn hex_value(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Percent-decode `%XX` sequences.  Browsers occasionally send encoded
/// paths; without decoding, `/E030%0A` would miss the registry lookup.
///
/// Returns a [`Cow`]: a path with no `%` — the overwhelmingly common request —
/// is handed back borrowed with zero allocation, and only genuinely encoded
/// paths pay for the output buffer.
fn percent_decode(s: &str) -> CompactString {
    if memchr::memchr(b'%', s.as_bytes()).is_none() {
        return CompactString::from(s);
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(hi), Some(lo)) = (hex_value(bytes[i + 1]), hex_value(bytes[i + 2]))
        {
            out.push((hi << 4) | lo);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    // HTTP paths are almost always ASCII after `%XX` decoding, so the common
    // case takes `String::from_utf8`'s zero-allocation `Ok` path; the lossy
    // fallback runs only on genuinely malformed UTF-8.
    match String::from_utf8(out) {
        Ok(s) => CompactString::from(s),
        Err(e) => CompactString::from(String::from_utf8_lossy(e.as_bytes()).as_ref()),
    }
}

fn build_index_page() -> &'static CachedPage {
    // The index is a pure function of the static CODE_TABLE, so build it
    // once and hand out a `'static` reference per request — far cheaper than
    // re-escaping 452 codes and re-formatting every entry on each `GET /`.
    static INDEX: std::sync::OnceLock<CachedPage> = std::sync::OnceLock::new();
    INDEX.get_or_init(|| CachedPage::build("200 OK", "", build_index_page_uncached()))
}

fn build_index_page_uncached() -> String {
    let groups = crate::diagnostics::error_code::code_entries_by_category();
    // reserve an exact-ish capacity from the metadata (code + title
    // bytes plus fixed markup) rather than a loose per-entry estimate, and
    // `write!` straight into `body` instead of `format!`ing a transient
    // `String` per entry.
    const FIXED_PER_ENTRY: usize = 96;
    let approx: usize = groups
        .iter()
        .map(|(cat, entries)| {
            cat.label().len()
                + entries
                    .iter()
                    .map(|e| e.code.len() + e.title.len() + FIXED_PER_ENTRY)
                    .sum::<usize>()
        })
        .sum();
    let mut body = String::with_capacity(approx + 4096);
    body.push_str(&page_head("Ponent error codes", INDEX_CSS));
    body.push_str("<h1>Ponent error codes</h1>\n");
    for (cat, entries) in groups {
        swrite!(body, "<h2>{}</h2>\n<ul>\n", escape_html(&cat.label()));
        for entry in entries {
            let code = escape_html(entry.code);
            let title = escape_html(entry.title);
            swrite!(
                body,
                "<li><a href=\"/{code}\"><span class=\"code\">{code}</span></a> — \
                 <span class=\"title\">{title}</span></li>\n",
            );
        }
        body.push_str("</ul>\n");
    }
    body.push_str("</body>\n</html>\n");
    body
}

fn build_not_found_page(code: &str) -> String {
    let mut out = page_head("Not found — Ponent", "");
    out.push_str("<h1>404 — not found</h1>\n<p>No error code <code>");
    // Filter control characters and escape HTML in one pass straight into
    // the destination — no intermediate `sanitized` String and no collected
    // positions for this untrusted-input path.
    let echo = match code.char_indices().nth(256) {
        Some((cut, _)) => std::borrow::Cow::Owned(format!("{}…", &code[..cut])),
        None => std::borrow::Cow::Borrowed(code),
    };
    write_html_escaped_filtered(&mut out, &echo);
    out.push_str(
        "</code> is registered.</p>\n\
         <p><a href=\"/\">Back to the index</a></p>\n\
         </body>\n</html>\n",
    );
    out
}

/// Append `input` to `out`, dropping control characters and HTML-escaping the
/// five text-node specials, in one pass.
fn write_html_escaped_filtered(out: &mut String, input: &str) {
    for ch in input.chars() {
        if ch.is_control() {
            continue;
        }
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
}

/// Entries admitted across *all* 404-cache shards.
///
/// The memory bound is global: admission reserves a slot here rather than
/// checking a fixed per-shard cap, so capacity follows demand under skew
/// instead of a hot shard evicting while idle shards hold empty slots.  It lives
/// at module scope (not inside [`not_found_page_cached`]) so the reservation
/// protocol and its stress test observe the same counter.
static NOT_FOUND_CACHE_TOTAL: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(all(test, not(feature = "shuttle")))]
fn not_found_cache_entry_count() -> usize {
    NOT_FOUND_CACHE_TOTAL.load(std::sync::atomic::Ordering::Relaxed)
}

/// Serve a cached 404 page for `code`.
///
/// Mirrors [`explain_page_cached`]: the page is a pure function of `code`,
/// so each distinct path's page is built at most once for the lifetime of
/// the persistent server.  Without the cache a client hammering `/a`, `/b`,
/// … forces an unbounded rebuild loop (`format!` + `escape_html` per
/// request); with it, a repeat request is an `Arc<str>` refcount bump.
///
/// The map is capped because distinct 404 paths are unbounded.  Past
/// [`NOT_FOUND_CACHE_CAP`] entries the old code called `clear()`, so an
/// attacker cycling paths evicted *every* cached page and forced all of them
/// to rebuild.  Evict one arbitrary (hash-order) victim instead: memory stays
/// bounded, a flood can only displace one entry per new key, and the common
/// hot pages survive.
fn not_found_page_cached(code: &str) -> std::sync::Arc<OwnedPage> {
    const NOT_FOUND_CACHE_CAP: usize = 1024;
    // [4]11: shard the 404 cache.  This is the adversarial path by design
    // (floods of distinct keys), so a single mutex serialized every miss and
    // every hit.  Sixteen independent map+CLOCK-ring shards make contention
    // proportional to collisions, not to total traffic.
    const SHARDS: usize = 16;
    const PER_SHARD_CAP: usize = NOT_FOUND_CACHE_CAP / SHARDS;
    // CLOCK (second-chance) admission instead of evicting an arbitrary
    // map key.  A cache hit sets the referenced bit; eviction sweeps the ring,
    // clearing bits and skipping referenced entries.
    struct Entry {
        key: CompactString,
        page: std::sync::OnceLock<std::sync::Arc<OwnedPage>>,
        referenced: std::sync::atomic::AtomicBool,
        live: std::sync::atomic::AtomicBool,
    }
    struct Shard {
        map: rustc_hash::FxHashMap<CompactString, std::sync::Arc<Entry>>,
        ring: std::collections::VecDeque<std::sync::Arc<Entry>>,
        /// Admission window: keys observed once, **with the page that first
        /// observation already built**.  A key must be seen twice before it may enter
        /// `map`, so a flood of one-hit paths (broken links, favicon probes) cannot
        /// displace pages that are actually requested repeatedly.
        ///
        /// A map, not a deque: the old `iter().position(..)` was O(window) on the
        /// adversarial path (every request, hit or miss), and the strike-1 page was
        /// dropped and rebuilt on strike 2 — two full page builds per admitted key.
        seen: rustc_hash::FxHashMap<CompactString, std::sync::Arc<OwnedPage>>,
        /// FIFO order for evicting `seen`, bounded exactly like it.  A key admitted (or
        /// re-struck) since it was queued is simply absent and gets skipped.
        seen_order: std::collections::VecDeque<CompactString>,
    }
    /// Per-shard size of the two-strike admission window.
    const SEEN_CAP: usize = PER_SHARD_CAP;
    static CACHE: std::sync::OnceLock<Box<[std::sync::Mutex<Shard>]>> = std::sync::OnceLock::new();
    let build = |code: &str| {
        let body = build_not_found_page(code);
        let head = render_head("404 Not Found", "", body.len());
        OwnedPage {
            head: head_parts_owned(head),
            body: std::sync::Arc::from(body),
        }
    };
    // Over-long keys are bounded in the body and still admitted; the cache cap
    // bounds entry count, and the reflected key is truncated to 256 chars.
    let shards = CACHE.get_or_init(|| {
        (0..SHARDS)
            .map(|_| {
                std::sync::Mutex::new(Shard {
                    map: rustc_hash::FxHashMap::default(),
                    ring: std::collections::VecDeque::new(),
                    seen: rustc_hash::FxHashMap::default(),
                    seen_order: std::collections::VecDeque::new(),
                })
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    });
    let shard_idx = {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        code.hash(&mut hasher);
        (hasher.finish() as usize) % SHARDS
    };
    let mut guard = shards[shard_idx].lock().unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = guard.map.get(code) {
        entry
            .referenced
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(page) = entry.page.get() {
            return std::sync::Arc::clone(page);
        }
        let entry = std::sync::Arc::clone(entry);
        drop(guard);
        return std::sync::Arc::clone(entry.page.get_or_init(|| std::sync::Arc::new(build(code))));
    }
    // Two-strike admission.  The first observation records the key in the
    // bounded window and serves the page *without* caching it; the second
    // observation admits.  Repeatedly requested paths — the ones a cache
    // actually helps — therefore survive a flood of unique ones, which under
    // the previous one-strike policy could cycle the whole ring.
    let key = CompactString::from(code);
    // Strike 2 carries the page strike 1 built, so admission publishes it instead of
    // rebuilding.  (The old code built it, dropped it, and built it again.)
    let window_page = match guard.seen.remove(&key) {
        Some(page) => page,
        None => {
            // First observation: build (it must be served anyway), keep it for the
            // possible second strike, and serve it uncached.
            let page = std::sync::Arc::new(build(code));
            if guard.seen.len() >= SEEN_CAP {
                // Evict the oldest window entry; one that was admitted since it was
                // queued is already gone from the map, so keep popping until one dies.
                while let Some(old) = guard.seen_order.pop_front() {
                    if guard.seen.remove(&old).is_some() {
                        break;
                    }
                }
            }
            guard.seen.insert(key.clone(), std::sync::Arc::clone(&page));
            guard.seen_order.push_back(key.clone());
            drop(guard);
            return page;
        }
    };
    // ── Global admission ────────────────────────────────────────────────
    // The shard mutex protects one shard, so the previous
    // `if TOTAL.load() >= CAP { evict }; …insert…; TOTAL.fetch_add(1)`
    // sequence was not transactional: two threads in different shards could
    // both observe `TOTAL < CAP`, both skip eviction, and both insert,
    // overshooting the advertised bound by one per racing pair.  `Relaxed` is
    // fine for counter arithmetic; it cannot make check-then-increment atomic.
    //
    // Reserve a global slot with a CAS *before* publishing the entry, and give
    // the reservation back when eviction cannot free one.  Every exit path
    // either publishes (ownership of the slot transfers to the cache) or lets
    // `SlotReservation::drop` subtract it again.
    fn try_reserve() -> bool {
        let mut cur = NOT_FOUND_CACHE_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
        loop {
            if cur >= NOT_FOUND_CACHE_CAP {
                return false;
            }
            match NOT_FOUND_CACHE_TOTAL.compare_exchange_weak(
                cur,
                cur + 1,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => cur = actual,
            }
        }
    }
    /// Release a reserved-but-unpublished slot (panic or unsuccessful
    /// publish), so a failed admission never leaks capacity.
    struct SlotReservation(bool);
    impl Drop for SlotReservation {
        fn drop(&mut self) {
            if self.0 {
                NOT_FOUND_CACHE_TOTAL.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    // CLOCK sweep, then (if every entry is referenced) the oldest live entry;
    // true iff an entry was actually removed and its global slot released.
    // A nested `fn` rather than a closure so `evict_from_other` can call it.
    fn evict_one(guard: &mut Shard) -> bool {
        // Bound the sweep per shard; if every entry is referenced, evict the
        // hand's candidate after two full rotations.
        for _ in 0..PER_SHARD_CAP.saturating_mul(2) {
            let Some(entry) = guard.ring.pop_front() else {
                break;
            };
            if !entry.live.load(std::sync::atomic::Ordering::Relaxed) {
                continue;
            }
            if entry
                .referenced
                .swap(false, std::sync::atomic::Ordering::Relaxed)
            {
                guard.ring.push_back(entry);
            } else {
                entry
                    .live
                    .store(false, std::sync::atomic::Ordering::Relaxed);
                if guard
                    .map
                    .get(&entry.key)
                    .is_some_and(|current| std::sync::Arc::ptr_eq(current, &entry))
                {
                    guard.map.remove(&entry.key);
                    NOT_FOUND_CACHE_TOTAL.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                }
                return true;
            }
        }
        // Nothing evictable in *this* shard while the budget is full: take the
        // oldest live entry.  A shard that cannot reclaim only serves the page
        // uncached — but never exceeds the global bound.
        match guard.ring.pop_front() {
            Some(entry) => {
                entry
                    .live
                    .store(false, std::sync::atomic::Ordering::Relaxed);
                let hit = guard
                    .map
                    .get(&entry.key)
                    .is_some_and(|current| std::sync::Arc::ptr_eq(current, &entry));
                if hit {
                    guard.map.remove(&entry.key);
                    NOT_FOUND_CACHE_TOTAL.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                }
                hit
            }
            None => false,
        }
    }
    /// Reclaim one slot from a shard *other* than ours.
    ///
    /// Only called with our own shard unlocked, so at most one shard lock is
    /// held at a time — no lock ordering, no deadlock.  This closes the gap
    /// where a shard that happened to fill first served uncached while other
    /// shards still held evictable entries, contradicting the global-budget
    /// policy.
    fn evict_from_other(shards: &[std::sync::Mutex<Shard>], me: usize) -> bool {
        for (i, shard) in shards.iter().enumerate() {
            if i == me {
                continue;
            }
            let mut guard = shard.lock().unwrap_or_else(|e| e.into_inner());
            if evict_one(&mut guard) {
                return true;
            }
        }
        false
    }
    // A freed slot can be stolen by a concurrent admission, so retry a bounded
    // number of times instead of assuming one eviction is enough.
    let mut reservation = SlotReservation(false);
    let mut admitted = false;
    for _ in 0..3 {
        if try_reserve() {
            reservation = SlotReservation(true);
            admitted = true;
            break;
        }
        if evict_one(&mut guard) {
            continue;
        }
        drop(guard);
        let freed = evict_from_other(shards, shard_idx);
        guard = shards[shard_idx].lock().unwrap_or_else(|e| e.into_inner());
        // Another thread may have admitted our key while we were unlocked: it
        // is in the map now, so serve it uncached instead of publishing a
        // duplicate entry for the same key.
        if guard.map.contains_key(code) {
            drop(guard);
            return window_page;
        }
        if !freed {
            break;
        }
    }
    if !admitted {
        drop(guard);
        return window_page;
    }
    let entry = std::sync::Arc::new(Entry {
        key: key.clone(),
        // Pre-filled with the page strike 1 built: `get_or_init` below would otherwise
        // rebuild it, so admission paid two full page builds per admitted key.
        page: {
            let cell = std::sync::OnceLock::new();
            // Fresh OnceLock: set always succeeds; unreachable if the invariant breaks.
            let _ = cell.set(window_page).unwrap_or_else(|_| unreachable!("OnceLock was just constructed"));
            cell
        },
        referenced: std::sync::atomic::AtomicBool::new(true),
        live: std::sync::atomic::AtomicBool::new(true),
    });
    guard.map.insert(key, std::sync::Arc::clone(&entry));
    guard.ring.push_back(std::sync::Arc::clone(&entry));
    // The published entry now accounts the reserved slot.
    reservation.0 = false;
    drop(reservation);
    drop(guard);
    std::sync::Arc::clone(entry.page.get_or_init(|| std::sync::Arc::new(build(code))))
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    /// Every style variant the page builders can pass must be in the CSP hash
    /// set: a variant that is not compiles and ships, and the browser then
    /// rejects the `<style>` block with no server-side signal (the page renders
    /// unstyled).  `page_head`'s debug assert is the other half of this pin.
    #[test]
    fn csp_hashes_cover_every_page_style_variant() {
        use base64::Engine as _;
        use sha2::{Digest, Sha256};
        let css = crate::diagnostics::emitter::stylesheet::dark_css();
        let src = csp_style_src();
        for extra in KNOWN_EXTRA_CSS {
            let digest = Sha256::digest(page_style(css, extra).as_bytes());
            let b64 = base64::engine::general_purpose::STANDARD.encode(digest);
            assert!(
                src.contains(&b64),
                "style variant {extra:?} is not covered by the CSP hash set"
            );
        }
    }

    /// Golden values for the civil-date arithmetic: the only test before this
    /// pinned prefix/suffix shape, so every month could have been off by one and
    /// the suite would still pass.
    #[test]
    fn http_date_goldens_pin_the_civil_arithmetic() {
        assert_eq!(format_http_date(0), "Thu, 01 Jan 1970 00:00:00 GMT");
        assert_eq!(
            format_http_date(1_700_000_000),
            "Tue, 14 Nov 2023 22:13:20 GMT"
        );
        // Leap day in a leap year (2024-02-29 12:00:00 UTC).
        assert_eq!(
            format_http_date(1_709_208_000),
            "Thu, 29 Feb 2024 12:00:00 GMT"
        );
        // Weekday roll-over across a month end (2023-12-31 → Sunday).
        assert_eq!(
            format_http_date(1_704_067_199),
            "Sun, 31 Dec 2023 23:59:59 GMT"
        );
    }

    /// routing must not act on a truncated first read.  A 600-byte
    /// request line (long `?` query) has no `\n` in the 512-byte probe, so
    /// the route check must decline and let `drain_request` finish reading
    /// instead of 404ing the truncated path.
    #[test]
    fn incomplete_request_line_is_not_routed() {
        let mut long = Vec::from(b"GET /E030?".as_slice());
        long.extend(std::iter::repeat_n(b'q', 600));
        assert!(
            complete_request_target(&long).is_none(),
            "a request line without a newline must not be routed"
        );
        long.extend_from_slice(b" HTTP/1.1\r\nHost: x\r\n\r\n");
        let expected = format!("/E030?{}", "q".repeat(600));
        assert_eq!(complete_request_target(&long), Some(expected.as_str()));
    }

    #[test]
    fn test_unknown_code_is_invalid_input_not_panic() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let err = serve_explain(&listener, "ZZZ").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains("unknown error code"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn percent_decode_accepts_only_real_hex_digits() {
        // `u8::from_str_radix` accepts a leading `+`; `%+F` must stay literal
        // instead of decoding to byte 0x0F.
        assert_eq!(percent_decode("%+F").as_str(), "%+F");
        assert_eq!(percent_decode("%GG").as_str(), "%GG");
        assert_eq!(percent_decode("%4").as_str(), "%4");
        assert_eq!(percent_decode("%2F").as_str(), "/");
        assert_eq!(percent_decode("%45%30%33%30").as_str(), "E030");
        assert_eq!(percent_decode("/plain").as_str(), "/plain");
    }

    #[test]
    fn dispatch_rejects_non_get_with_allow_header() {
        let response = dispatch_request("POST /E030 HTTP/1.1");
        assert_eq!(response.status, "405 Method Not Allowed");
        assert_eq!(response.extra_headers, "Allow: GET, HEAD\r\n");
        assert!(response.body().contains("only accepts"));
    }

    #[test]
    fn head_error_responses_are_bodyless() {
        for line in ["HEAD /E030 HTTP/2", "HEAD E030 HTTP/1.1"] {
            let response = dispatch_request(line);
            assert_eq!(response.status, "400 Bad Request", "line {line:?}");
            assert!(
                response.body().is_empty(),
                "HEAD 400 must not carry a body: {line:?}"
            );
        }
    }

    #[test]
    fn dispatch_rejects_malformed_request_lines() {
        for line in [
            "",
            "GET",
            "GET /",
            "GET / HTTP/1.1 extra",
            "GET E030 HTTP/1.1",
            "GET /E030 HTTP/2",
        ] {
            let response = dispatch_request(line);
            assert_eq!(
                response.status, "400 Bad Request",
                "request line {line:?} should be rejected"
            );
        }
    }

    #[test]
    fn dispatch_strips_exactly_one_slash() {
        assert_eq!(dispatch_request("GET / HTTP/1.1").status, "200 OK");
        assert_eq!(dispatch_request("GET /E030 HTTP/1.1").status, "200 OK");
        assert_eq!(dispatch_request("GET /E030/ HTTP/1.1").status, "200 OK");
        assert_eq!(
            dispatch_request("GET //E030 HTTP/1.1").status,
            "404 Not Found"
        );
        assert_eq!(
            dispatch_request("GET /E030?view=full HTTP/1.1").status,
            "200 OK"
        );
        assert_eq!(
            dispatch_request("GET /%45%30%33%30 HTTP/1.1").status,
            "200 OK"
        );
    }

    #[test]
    fn escape_html_covers_every_character() {
        assert_eq!(escape_html("<&>\"'").as_ref(), "&lt;&amp;&gt;&quot;&#39;");
        assert_eq!(escape_html("plain").as_ref(), "plain");
    }

    #[test]
    fn index_page_has_viewport_theme_toggle_and_css_vars() {
        let page = build_index_page();
        let page = page.body.as_ref();
        assert!(page.contains(r#"<meta name="viewport""#));
        assert!(page.contains(r#"id="theme-toggle""#));
        assert!(page.contains("var(--bg)"));
        assert!(page.contains("Ponent error codes"));
    }

    #[test]
    fn not_found_page_uses_shared_theme() {
        let page = build_not_found_page("<bad>");
        assert!(page.contains("var(--bg)"));
        assert!(page.contains("&lt;bad&gt;"));
        assert!(page.contains(r#"id="theme-toggle""#));
    }

    /// Regression: the shard mutex protects one shard, so admission has to be
    /// transactional.  Before the reservation protocol two threads in different
    /// shards could both observe `TOTAL < CAP`, both skip eviction, and both
    /// publish, exceeding the advertised bound by one per racing pair.  This
    /// hammers every shard from several threads and pins the global bound.
    #[test]
    fn concurrent_not_found_admissions_respect_the_global_cap() {
        // `NOT_FOUND_CACHE_CAP` is function-local; keep this in sync with it.
        const CAP: usize = 1024;
        let keys: Vec<String> = (0..2048).map(|i| format!("E{i:04}")).collect();
        let keys = std::sync::Arc::new(keys);
        let threads: Vec<_> = (0..6)
            .map(|_| {
                let keys = std::sync::Arc::clone(&keys);
                std::thread::spawn(move || {
                    // Rounds give the two-strike window a chance to admit each
                    // key before it ages out, so real admissions (and therefore
                    // evictions) race rather than hits short-circuiting.
                    for _ in 0..3 {
                        for key in keys.iter() {
                            swallow!(not_found_page_cached(key));
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let count = not_found_cache_entry_count();
        assert!(count <= CAP, "global 404 cache exceeded its cap: {count}");
        // The counter must also agree with the live entries it claims to
        // account for: a leaked reservation would inflate it.
        assert!(count > 0, "the stress workload admitted nothing");
    }

    /// A request body announced by `Content-Length` must be read before the
    /// response socket is shut down; otherwise the kernel sends RST for the
    /// unread bytes and a client's `read_to_end` fails.
    #[test]
    fn serve_page_drains_content_length_body() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let body = "<html>hi</html>".to_string();

        let h = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            serve_page(&mut s, &body).unwrap();
        });

        let mut client = TcpStream::connect(addr).unwrap();
        client
            .write_all(b"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello")
            .unwrap();
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();
        assert!(response.starts_with(b"HTTP/1.1 200 OK"));
        h.join().unwrap();
    }

    #[test]
    fn persistent_server_answers_post_with_405() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            swallow!(serve_explain_persistent(&listener));
        });

        let mut client = TcpStream::connect(addr).unwrap();
        client
            .write_all(b"POST /E030 HTTP/1.1\r\nContent-Length: 0\r\n\r\n")
            .unwrap();
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();
        assert!(
            response.starts_with(b"HTTP/1.1 405 Method Not Allowed"),
            "response: {}",
            String::from_utf8_lossy(&response)
        );
        assert!(
            response
                .windows(b"Allow: GET, HEAD".len())
                .any(|w| w == b"Allow: GET, HEAD"),
            "response: {}",
            String::from_utf8_lossy(&response)
        );
    }

    /// Thread-per-connection: several overlapping client connections must all
    /// be served — a serialized accept loop made each connection pay the
    /// previous one's read timeout.  The shared caches (`Mutex`-guarded page
    /// caches, `OnceLock` index) are thread-safe, so parallel requests each
    /// get the correct page.
    #[test]
    fn persistent_server_serves_concurrent_connections() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            swallow!(serve_explain_persistent(&listener));
        });

        let mut handles = Vec::new();
        for _ in 0..4 {
            handles.push(std::thread::spawn(move || {
                let mut c = TcpStream::connect(addr).unwrap();
                c.write_all(b"GET /E030 HTTP/1.1\r\nHost: x\r\n\r\n")
                    .unwrap();
                let mut buf = Vec::new();
                swallow!(c.read_to_end(&mut buf));
                buf.starts_with(b"HTTP/1.1 200 OK")
            }));
        }
        for h in handles {
            assert!(h.join().unwrap());
        }
    }

    #[test]
    fn persistent_server_accepts_lf_only_request_line() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            swallow!(serve_explain_persistent(&listener));
        });

        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        client
            .write_all(b"GET /E030 HTTP/1.1\nHost: x\n\n")
            .unwrap();
        let mut response = Vec::new();
        let mut chunk = [0u8; 4096];
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            match client.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    response.extend_from_slice(&chunk[..n]);
                    if response.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        assert!(
            response.starts_with(b"HTTP/1.1 200 OK"),
            "LF-only request line was rejected: {:?}",
            &response[..response.len().min(40)]
        );
    }

    #[test]
    fn persistent_server_reuses_keep_alive_connection() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            swallow!(serve_explain_persistent(&listener));
        });

        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let request = b"GET /E030 HTTP/1.1\r\nHost: x\r\nConnection: keep-alive\r\n\r\n";
        client.write_all(request).unwrap();

        let mut response = Vec::new();
        let mut chunk = [0u8; 4096];
        let deadline = Instant::now() + Duration::from_secs(3);
        let count = |buf: &[u8]| {
            buf.windows(b"HTTP/1.1 200 OK".len())
                .filter(|w| *w == b"HTTP/1.1 200 OK")
                .count()
        };
        // First response: wait for its head, then send the second request.
        while Instant::now() < deadline && count(&response) < 1 {
            match client.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => response.extend_from_slice(&chunk[..n]),
            }
        }
        assert_eq!(count(&response), 1, "first request was not served");
        client.write_all(request).unwrap();
        while Instant::now() < deadline && count(&response) < 2 {
            match client.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => response.extend_from_slice(&chunk[..n]),
            }
        }
        assert_eq!(
            count(&response),
            2,
            "keep-alive connection did not serve the second request"
        );
    }

    #[test]
    fn persistent_server_sheds_at_connection_cap_with_503() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            swallow!(serve_explain_persistent(&listener));
        });
        // Hold the reactor at its connection cap.  Each holder writes **one byte** of an
        // incomplete request line: a connected socket that sends nothing sits in the
        // reactor's first-byte window, so the cap only filled after that timeout expired
        // and the 50 ms sleep below was a race against it.  A partial line is unambiguously
        // "Reading", which is the state the cap counts.
        let mut idle = Vec::new();
        for _ in 0..MAX_CONCURRENT_CONNECTIONS {
            let mut holder = TcpStream::connect(addr).unwrap();
            holder.write_all(b" ").unwrap();
            idle.push(holder);
        }
        std::thread::sleep(Duration::from_millis(50));
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        client
            .write_all(b"GET /E030 HTTP/1.1\r\nHost: x\r\n\r\n")
            .unwrap();
        let mut response = Vec::new();
        let mut chunk = [0u8; 1024];
        while let Ok(n) = client.read(&mut chunk) {
            if n == 0 {
                break;
            }
            response.extend_from_slice(&chunk[..n]);
            if response.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        assert!(
            response.starts_with(b"HTTP/1.1 503"),
            "shed path must deliver the 503, got {:?}",
            &response[..response.len().min(32)]
        );
        drop(idle);
    }

    #[test]
    fn thread_per_connection_path_serves_a_request_on_linux() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let server = std::thread::spawn(move || {
            handle_persistent_connection(&mut server, &AtomicUsize::new(0))
        });
        client
            .write_all(b"GET /E030 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut response = Vec::new();
        // `read_to_end` rides the server's close; the 2 s timeout keeps a
        // regression from wedging the suite.
        swallow!(client.read_to_end(&mut response));
        server.join().unwrap().unwrap();
        assert!(
            response.starts_with(b"HTTP/1.1 200 OK"),
            "thread path response: {:?}",
            &response[..response.len().min(40)]
        );
    }

    /// Shared network helper: one request, one-shot server join.
    /// Returns the response body and any read error so callers can assert on both.
    fn one_shot_serve(code: &str) -> (Vec<u8>, Option<std::io::Error>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let code_owned = code.to_string();
        let server = std::thread::spawn(move || {
            // Serve with the owned string; `serve_explain` borrows it for the
            // duration of the call, which is fine here.
            serve_explain(&listener, &code_owned)
        });
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        client
            .write_all(
                format!("GET /{code} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").as_bytes(),
            )
            .unwrap();
        let mut response = Vec::new();
        let read_err = client.read_to_end(&mut response).err();
        server.join().unwrap().unwrap();
        (response, read_err)
    }

    #[test]
    #[cfg(unix)]
    fn one_shot_restores_caller_nonblocking_mode() {
        for initial_nonblocking in [true, false] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(initial_nonblocking).unwrap();
            let probe = listener.try_clone().unwrap();
            let addr = listener.local_addr().unwrap();
            let server = std::thread::spawn(move || serve_explain(&listener, "E030"));
            let mut client = TcpStream::connect(addr).unwrap();
            client
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            client
                .write_all(b"GET /E030 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
                .unwrap();
            let mut response = Vec::new();
            swallow!(client.read_to_end(&mut response));
            server.join().unwrap().unwrap();
            assert_eq!(
                socket2::SockRef::from(&probe).nonblocking().unwrap(),
                initial_nonblocking,
                "one-shot server must restore the caller's mode"
            );
        }
    }

    #[test]
    fn one_shot_server_smoke_test() {
        // Winsock has no way to query the nonblocking bit, so the
        // nonblocking-restore assertion is unix-only (see above).
        // On all platforms we verify the server responds correctly.
        let (response, read_err) = one_shot_serve("E030");
        assert!(
            response.starts_with(b"HTTP/1.1 200 OK"),
            "server should respond (read_err={read_err:?}): {:?}",
            &response[..response.len().min(40)],
        );
        assert!(
            read_err.is_none(),
            "read should complete without error: {read_err:?}"
        );
    }

    #[test]
    fn reactor_rejects_request_line_over_8_kib() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            swallow!(serve_explain_persistent(&listener));
        });
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut request = b"GET /".to_vec();
        request.extend(std::iter::repeat(b'a').take(9 * 1024));
        request.extend_from_slice(b" HTTP/1.1\r\nHost: x\r\n\r\n");
        swallow!(client.write_all(&request));
        let mut response = Vec::new();
        swallow!(client.read_to_end(&mut response));
        assert!(
            !response.starts_with(b"HTTP/1.1 200"),
            "oversized request line must not be served"
        );
    }

    #[test]
    fn zero_copy_output_survives_short_reads() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            swallow!(serve_explain_persistent(&listener));
        });
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        client
            .write_all(b"GET /NO_SUCH_CODE HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .unwrap();
        // One byte per read starves the socket deliberately, exercising the
        // write cursor across partial `write_vectored` progress.
        let mut response = Vec::new();
        let mut byte = [0u8; 1];
        while let Ok(1) = client.read(&mut byte) {
            response.push(byte[0]);
            if response.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        assert!(
            response.starts_with(b"HTTP/1.1 404"),
            "short-read response: {:?}",
            &response[..response.len().min(40)]
        );
    }

    /// Security: the explain page is an HTTP endpoint, so a listener bound
    /// to a non-loopback (wildcard) address must be refused rather than
    /// served to the whole network.
    #[test]
    fn non_loopback_listener_is_rejected() {
        let listener = match TcpListener::bind("0.0.0.0:0") {
            Ok(l) => l,
            // Sandboxes that forbid wildcard binds cannot exercise this; the
            // check in `serve_explain` is unconditional either way.
            Err(_) => return,
        };
        let err = serve_explain(&listener, "E030").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains("loopback"),
            "unexpected message: {err}"
        );
    }

    /// A client that dribbles a byte and then stalls must not hold the drain
    /// loop for the old ~50 s worst case; the wall-clock budget caps it.
    #[test]
    fn serve_page_does_not_hang_on_slow_client() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let body = "<html>hi</html>".to_string();

        let h = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            serve_page(&mut s, &body).unwrap();
        });

        let mut client = TcpStream::connect(addr).unwrap();
        client.write_all(b"G").unwrap();
        let t0 = Instant::now();
        h.join().unwrap();
        assert!(
            t0.elapsed() < Duration::from_secs(3),
            "serve_page hung for {:?}",
            t0.elapsed()
        );
    }

    /// Regression: an HTTP request whose `\r\n\r\n` terminator is split
    /// across two reads must be detected via the carry buffer, ending the
    /// drain as soon as it completes — not after the 500 ms read timeout.
    /// (The old per-chunk `windows(4)` scan never saw the split.)
    #[test]
    fn serve_page_drains_split_terminator_without_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let body = "<html>hi</html>".to_string();

        let h = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            serve_page(&mut s, &body).unwrap();
        });

        let mut client = TcpStream::connect(addr).unwrap();
        // First read gets everything up to (and including) the lone `\r`;
        // the terminator completes only in the second read.
        client.write_all(b"GET / HTTP/1.1\r").unwrap();
        std::thread::sleep(Duration::from_millis(50));
        client.write_all(b"\nHost: x\r\n\r\n").unwrap();

        let mut response = Vec::new();
        let t0 = Instant::now();
        swallow!(client.read_to_end(&mut response));
        // The old code stalls for the 500 ms read timeout before draining;
        // the carry buffer must end the drain as soon as the split
        // terminator completes (~50 ms + scheduling overhead).
        assert!(
            t0.elapsed() < Duration::from_millis(250),
            "split terminator still stalled the drain for {:?}",
            t0.elapsed()
        );
        assert!(response.starts_with(b"HTTP/1.1 200 OK"));
        h.join().unwrap();
    }

    /// Regression: a browser's speculative preconnect (a TCP connection that
    /// sends no request) must not consume the single page this server waits
    /// to serve.  The preconnect is closed and the accept loop keeps waiting
    /// for the socket that actually carries the request.
    #[test]
    fn preconnect_without_request_does_not_consume_the_page() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let h = std::thread::spawn(move || serve_explain(&listener, "E030"));

        // Speculative preconnect: open and send nothing.
        let pre = TcpStream::connect(addr).unwrap();
        std::thread::sleep(Duration::from_millis(200));

        // The real request arrives on a second connection.
        let mut real = TcpStream::connect(addr).unwrap();
        real.write_all(b"GET /E030 HTTP/1.1\r\nHost: x\r\n\r\n")
            .unwrap();
        let mut response = Vec::new();
        swallow!(real.read_to_end(&mut response));
        assert!(
            response.starts_with(b"HTTP/1.1 200 OK"),
            "response: {response:?}"
        );

        h.join().unwrap().unwrap();
        drop(pre);
    }

    #[test]
    fn canonical_path_code_keeps_percent_encoded_slash_distinct() {
        assert_eq!(canonical_path_code("/E030").as_deref(), Some("E030"));
        assert_eq!(
            canonical_path_code("/%45%30%33%30").as_deref(),
            Some("E030")
        );
        // An encoded slash must not become a path separator and alias `/E030/`.
        assert_eq!(canonical_path_code("/E030%2F"), None);
        assert_eq!(canonical_path_code("//E030"), None);
    }

    /// `decide` is the single source of "must we drain?" / "may we reuse?", so
    /// pin it against the predicates the backends used to spell out inline.  If
    /// this matrix and those formulas ever disagree, one backend would reuse a
    /// connection with an unread body (RST) or close one needlessly.
    #[test]
    fn decide_matches_the_inline_predicates_it_replaced() {
        for &has_te in &[false, true] {
            for &ambiguous in &[false, true] {
                for &malformed in &[false, true] {
                    for &oversized in &[false, true] {
                        for &close in &[false, true] {
                            for &content_length in &[None, Some(0usize), Some(5), Some(9)] {
                                for &already_read in &[0usize, 3, 5, 9, 20] {
                                    for &http11 in &[false, true] {
                                        let framing = HeaderFraming {
                                            content_length,
                                            has_transfer_encoding: has_te,
                                            ambiguous,
                                            malformed,
                                            oversized,
                                            connection_close: close,
                                        };
                                        let body = framing.body(already_read);
                                        // The two formulas the sites used.
                                        let old_close =
                                            close || !matches!(body, RequestBody::Empty);
                                        let old_keep =
                                            http11 && matches!(body, RequestBody::Empty) && !close;

                                        let d = framing.decide(already_read, http11);
                                        assert_eq!(
                                            d.must_close, old_close,
                                            "{framing:?} ar={already_read}"
                                        );
                                        assert_eq!(
                                            d.keep_alive_allowed, old_keep,
                                            "{framing:?} ar={already_read} http11={http11}"
                                        );
                                        assert_eq!(
                                            d.body_remaining,
                                            match body {
                                                RequestBody::Empty => Some(0),
                                                RequestBody::Fixed { remaining } => Some(remaining),
                                                RequestBody::Unframed => None,
                                            },
                                            "{framing:?} ar={already_read}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scan_framing_flags_transfer_encoding_and_ambiguous_lengths() {
        assert!(
            scan_framing(b"Transfer-Encoding: chunked\r\n\r\n", false).has_transfer_encoding,
            "chunked request must not be reusable as keep-alive"
        );
        assert!(
            scan_framing(b"Content-Length: 5\r\nContent-Length: 6\r\n\r\n", false).ambiguous,
            "conflicting Content-Length must be rejected"
        );
        assert!(
            scan_framing(b"Content-Length: +5\r\n\r\n", false).ambiguous,
            "signed Content-Length must be rejected"
        );
        assert!(
            scan_framing(
                b"Content-Length: 999999999999999999999999999999\r\n\r\n",
                false
            )
            .oversized,
            "a Content-Length beyond the body cap must be marked oversized"
        );
    }

    /// A colon-less header line must not silently become "no framing": if it was
    /// meant as `Content-Length 100`, treating the request as an unambiguous
    /// empty body would leave the body bytes in the stream (smuggling shape).
    /// The request line and the blank terminator are not header fields.
    #[test]
    fn scan_framing_marks_colonless_header_lines_as_malformed() {
        let head = b"GET /E030 HTTP/1.1\r\nHost: localhost\r\nContent-Length 100\r\n\r\n";
        let framing = scan_framing(head, true);
        assert!(
            framing.malformed,
            "colon-less header line must be malformed"
        );
        assert!(
            framing.content_length.is_none(),
            "the malformed line must not be parsed as a length"
        );
        assert_eq!(
            framing.body(0),
            RequestBody::Unframed,
            "a malformed head must never be reusable"
        );

        // Well-formed heads (including a bare LF flavour) stay clean.
        for head in [
            &b"GET /E030 HTTP/1.1\r\nHost: localhost\r\n\r\n"[..],
            &b"GET /E030 HTTP/1.1\nHost: localhost\n\n"[..],
            &b"GET /E030 HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n"[..],
        ] {
            assert!(!scan_framing(head, true).malformed, "head: {:?}", head);
        }
    }

    #[test]
    fn one_shot_keeps_fragmented_request_pending() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || serve_explain(&listener, "E030"));
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        client
            .write_all(b"GET /E030 HTTP/1.1\r\nHost: x\r\n")
            .unwrap();
        std::thread::sleep(Duration::from_millis(100));
        client.write_all(b"Connection: close\r\n\r\n").unwrap();
        let mut response = Vec::new();
        swallow!(client.read_to_end(&mut response));
        server.join().unwrap().unwrap();
        assert!(
            response.starts_with(b"HTTP/1.1 200"),
            "fragmented one-shot request was closed: {:?}",
            &response[..response.len().min(32)]
        );
    }

    #[test]
    fn one_shot_head_completes_without_body() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || serve_explain(&listener, "E030"));
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        client
            .write_all(b"HEAD /E030 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut response = Vec::new();
        swallow!(client.read_to_end(&mut response));
        server.join().unwrap().unwrap();
        assert!(
            response.starts_with(b"HTTP/1.1 200"),
            "one-shot HEAD failed: {:?}",
            &response[..response.len().min(32)]
        );
        let split = response
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .expect("complete response head");
        assert!(
            response[split + 4..].is_empty(),
            "HEAD response carried a body"
        );
    }

    #[test]
    fn one_shot_serves_see_also_after_target() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || serve_explain(&listener, "E030"));

        let mut first = TcpStream::connect(addr).unwrap();
        first
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        first
            .write_all(b"GET /E030 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut first_response = Vec::new();
        swallow!(first.read_to_end(&mut first_response));
        assert!(first_response.starts_with(b"HTTP/1.1 200"));

        // The one-shot server now keeps a short idle window alive so a
        // "See also" link to another registered code still resolves.
        let mut second = TcpStream::connect(addr).unwrap();
        second
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        second
            .write_all(b"GET /E001 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut second_response = Vec::new();
        swallow!(second.read_to_end(&mut second_response));
        server.join().unwrap().unwrap();
        assert!(
            second_response.starts_with(b"HTTP/1.1 200"),
            "See also page was not served: {:?}",
            &second_response[..second_response.len().min(32)]
        );
    }

    #[test]
    fn bounded_persistent_exits_after_idle() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            serve_explain_persistent_bounded(&listener, Duration::from_millis(200))
        });
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        client
            .write_all(b"GET /E030 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut response = Vec::new();
        swallow!(client.read_to_end(&mut response));
        server.join().unwrap().unwrap();
        assert!(response.starts_with(b"HTTP/1.1 200"));
    }

    #[test]
    fn reactor_continues_serving_while_another_connection_drains() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            serve_explain_persistent_bounded(&listener, Duration::from_millis(600))
        });

        let mut slow = TcpStream::connect(addr).unwrap();
        slow.write_all(b"GET /E030 HTTP/1.1\r\nHost: x\r\nContent-Length: 100000000\r\n\r\n")
            .unwrap();
        let mut slow_writer = slow.try_clone().unwrap();
        let writer = std::thread::spawn(move || {
            let chunk = [b'x'; 8192];
            for _ in 0..200 {
                if slow_writer.write_all(&chunk).is_err() {
                    break;
                }
            }
        });
        std::thread::sleep(Duration::from_millis(50));

        let mut fast = TcpStream::connect(addr).unwrap();
        fast.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
        fast.write_all(b"GET /E030 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut response = Vec::new();
        swallow!(fast.read_to_end(&mut response));
        swallow!(writer.join());
        swallow!(slow.shutdown(Shutdown::Both));
        swallow!(server.join());
        assert!(
            response.starts_with(b"HTTP/1.1 200"),
            "reactor was pinned by the draining connection: {:?}",
            String::from_utf8_lossy(&response[..response.len().min(32)])
        );
    }

    /// Both 400 exits on the one-shot path must be the *same* 400.
    ///
    /// A request whose first line is not valid UTF-8 took a separate exit that wrote the
    /// response directly, while a syntax failure went through `finish_response` — two
    /// policies for one status code, so a client could observe different headers (and a
    /// different keep-alive answer) purely from *why* its request was malformed.
    #[test]
    fn invalid_utf8_and_syntax_errors_share_one_bad_request_shape() {
        fn run(request: &[u8]) -> Vec<u8> {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let server = std::thread::spawn(move || serve_explain(&listener, "E030"));
            let mut client = TcpStream::connect(addr).unwrap();
            client
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            client.write_all(request).unwrap();
            swallow!(client.shutdown(std::net::Shutdown::Write));
            let mut response = Vec::new();
            swallow!(client.read_to_end(&mut response));
            swallow!(server.join().unwrap());
            response
        }
        /// The status line and headers, minus `Date` — that one changes between the two
        /// requests (second granularity), so comparing it would make this flaky rather
        /// than catch the policy divergence it is looking for.
        fn head_block(response: &[u8]) -> String {
            let end = response
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .unwrap_or(response.len());
            String::from_utf8_lossy(&response[..end])
                .lines()
                .filter(|line| !line.starts_with("Date:"))
                .collect::<Vec<_>>()
                .join("\n")
        }

        let utf8 = run(b"\xff\xfe / HTTP/1.1\r\nHost: x\r\n\r\n");
        let syntax = run(b"GET\x00 / HTTP/1.1\r\nHost: x\r\n\r\n");
        assert!(
            utf8.starts_with(b"HTTP/1.1 400"),
            "invalid UTF-8 must be a 400: {:?}",
            String::from_utf8_lossy(&utf8[..utf8.len().min(48)])
        );
        assert!(
            syntax.starts_with(b"HTTP/1.1 400"),
            "a NUL in the request line must be a 400: {:?}",
            String::from_utf8_lossy(&syntax[..syntax.len().min(48)])
        );
        assert_eq!(
            head_block(&utf8),
            head_block(&syntax),
            "the two 400 paths must not disagree on headers or connection policy"
        );
    }

    /// `Transfer-Encoding: identity` must not make the framing unprovable.
    ///
    /// `identity` is RFC 9112 §6.1's "no transfer coding" token.  Treating it like
    /// `chunked` sent every drain to the "wait for peer EOF" fallback, and a keep-alive
    /// client never closes — so each such request paid the full 1 s read budget.
    #[test]
    fn transfer_encoding_identity_keeps_the_framing_provable() {
        let identity = scan_framing(b"Transfer-Encoding: identity\r\nContent-Length: 0", false);
        assert!(
            !identity.has_transfer_encoding,
            "`identity` is not a transfer coding"
        );
        assert_eq!(identity.content_length, Some(0));
        assert!(
            scan_framing(b"Transfer-Encoding: chunked", false).has_transfer_encoding,
            "`chunked` must stay unprovable"
        );
        assert!(
            scan_framing(b"Transfer-Encoding: identity, chunked", false).has_transfer_encoding,
            "a mixed list still names a real coding"
        );
        assert!(
            scan_framing(b"Transfer-Encoding: Identity", false).has_transfer_encoding == false,
            "the token is case-insensitive"
        );
    }

    #[test]
    fn one_shot_serves_lf_only_request() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || serve_explain(&listener, "E030"));
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        client
            .write_all(b"GET /E030 HTTP/1.1\nHost: x\nConnection: close\n\n")
            .unwrap();
        let mut response = Vec::new();
        swallow!(client.read_to_end(&mut response));
        server.join().unwrap().unwrap();
        assert!(
            response.starts_with(b"HTTP/1.1 200"),
            "one-shot rejected an LF-only request: {:?}",
            &response[..response.len().min(32)]
        );
    }

    #[test]
    fn thread_path_serves_lf_only_request() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let server = std::thread::spawn(move || {
            handle_persistent_connection(&mut server, &AtomicUsize::new(0))
        });
        client
            .write_all(b"GET /E030 HTTP/1.1\nHost: x\nConnection: close\n\n")
            .unwrap();
        let mut response = Vec::new();
        swallow!(client.read_to_end(&mut response));
        server.join().unwrap().unwrap();
        assert!(
            response.starts_with(b"HTTP/1.1 200"),
            "thread path rejected an LF-only request: {:?}",
            &response[..response.len().min(32)]
        );
    }

    #[test]
    fn oversized_request_line_gets_431() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            swallow!(serve_explain_persistent(&listener));
        });
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut request = b"GET /".to_vec();
        request.extend(std::iter::repeat(b'a').take(9 * 1024));
        request.extend_from_slice(b" HTTP/1.1\r\nHost: x\r\n\r\n");
        swallow!(client.write_all(&request));
        let mut response = Vec::new();
        swallow!(client.read_to_end(&mut response));
        assert!(
            response.starts_with(b"HTTP/1.1 431"),
            "oversized request line must get 431, got {:?}",
            &response[..response.len().min(32)]
        );
    }

    #[test]
    fn short_request_line_with_large_header_is_served() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let server = std::thread::spawn(move || {
            handle_persistent_connection(&mut server, &AtomicUsize::new(0))
        });
        let mut request = b"GET /E030 HTTP/1.1\r\nHost: x\r\nCookie: ".to_vec();
        request.extend(std::iter::repeat(b'a').take(12 * 1024));
        request.extend_from_slice(b"\r\nConnection: close\r\n\r\n");
        client.write_all(&request).unwrap();
        let mut response = Vec::new();
        swallow!(client.read_to_end(&mut response));
        server.join().unwrap().unwrap();
        assert!(
            response.starts_with(b"HTTP/1.1 200"),
            "large header with short request line must be served: {:?}",
            &response[..response.len().min(32)]
        );
    }

    #[test]
    fn silent_overload_connection_gets_no_response() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            swallow!(serve_explain_persistent(&listener));
        });
        let mut idle = Vec::new();
        for _ in 0..MAX_CONCURRENT_CONNECTIONS {
            idle.push(TcpStream::connect(addr).unwrap());
        }
        std::thread::sleep(Duration::from_millis(50));
        let mut silent = TcpStream::connect(addr).unwrap();
        silent
            .set_read_timeout(Some(Duration::from_millis(250)))
            .unwrap();
        let mut byte = [0u8; 1];
        let read = silent.read(&mut byte);
        assert!(
            read.is_err() || matches!(read, Ok(0)),
            "silent overload connection must not receive a response"
        );
        drop(idle);
    }

    #[test]
    fn keep_alive_status_policy_is_shared() {
        for status in [
            "200 OK",
            "400 Bad Request",
            "404 Not Found",
            "405 Method Not Allowed",
        ] {
            assert!(response_allows_keep_alive(status), "{status}");
        }
        for status in [
            "500 Internal Server Error",
            "503 Service Unavailable",
            "431 ...",
        ] {
            assert!(!response_allows_keep_alive(status), "{status}");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn backends_agree_on_request_lifecycle() {
        fn thread_exchange(request: &[u8]) -> Vec<u8> {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let mut client = TcpStream::connect(addr).unwrap();
            client
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let (mut server, _) = listener.accept().unwrap();
            let server = std::thread::spawn(move || {
                handle_persistent_connection(&mut server, &AtomicUsize::new(0))
            });
            client.write_all(request).unwrap();
            let mut response = Vec::new();
            swallow!(client.read_to_end(&mut response));
            server.join().unwrap().unwrap();
            response
        }

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let reactor_addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            swallow!(serve_explain_persistent(&listener));
        });

        let mut large_header = b"GET /E030 HTTP/1.1\r\nHost: x\r\nCookie: ".to_vec();
        large_header.extend(std::iter::repeat(b'a').take(12 * 1024));
        large_header.extend_from_slice(b"\r\nConnection: close\r\n\r\n");

        let mut oversized = b"GET /".to_vec();
        oversized.extend(std::iter::repeat(b'a').take(9 * 1024));
        oversized.extend_from_slice(b" HTTP/1.1\r\nHost: x\r\n\r\n");

        let cases: [(&[u8], &[u8]); 8] = [
            (
                b"GET /E030 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
                b"200",
            ),
            (
                b"GET /E001 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
                b"200",
            ),
            (
                b"GET /E030 HTTP/1.1\nHost: x\nConnection: close\n\n",
                b"200",
            ),
            (
                b"GET /NO_SUCH_CODE HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
                b"404",
            ),
            (&oversized, b"431"),
            (&large_header, b"200"),
            // Empty header block: after the request line there is only the
            // single blank line, not a second terminator.
            (b"GET /E030 HTTP/1.0\r\n\r\n", b"200"),
            // Declared body must be consumed before the 405 is written; both
            // backends must agree on the status and not RST the response.
            (
                b"POST /E030 HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello",
                b"405",
            ),
        ];

        for (request, status) in cases {
            let mut reactor = TcpStream::connect(reactor_addr).unwrap();
            reactor
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            reactor.write_all(request).unwrap();
            let mut reactor_response = Vec::new();
            swallow!(reactor.read_to_end(&mut reactor_response));

            let thread_response = thread_exchange(request);
            assert!(
                reactor_response.starts_with(b"HTTP/1.1 ")
                    && thread_response.starts_with(b"HTTP/1.1 "),
                "backend produced no HTTP status"
            );
            let reactor_status = &reactor_response[9..12];
            let thread_status = &thread_response[9..12];
            assert_eq!(
                reactor_status,
                thread_status,
                "backends disagree for request: {:?}",
                String::from_utf8_lossy(&request[..request.len().min(32)])
            );
            assert_eq!(
                reactor_status,
                status,
                "unexpected status for case: {:?}",
                String::from_utf8_lossy(&request[..request.len().min(32)])
            );
            fn body(resp: &[u8]) -> &[u8] {
                resp.windows(4)
                    .position(|w| w == b"\r\n\r\n")
                    .map_or(&[], |i| &resp[i + 4..])
            }
            assert_eq!(
                body(&reactor_response),
                body(&thread_response),
                "backends disagree on body for request: {:?}",
                String::from_utf8_lossy(&request[..request.len().min(32)])
            );
        }
    }

    #[test]
    fn header_scan_enforces_shared_limits_and_empty_headers() {
        let request = b"GET / HTTP/1.0\r\n\r\n";
        let mut scan = HeaderScan::new();
        let mut buf = Vec::new();
        let mut last = HeaderProgress::Incomplete;
        for &byte in request {
            buf.push(byte);
            last = scan.scan(&buf);
        }
        assert_eq!(last, HeaderProgress::Complete(request.len()));

        let mut empty = HeaderScan::new();
        assert_eq!(
            empty.scan_after_request_line(b"\r"),
            HeaderProgress::Incomplete
        );
        assert_eq!(
            empty.scan_after_request_line(b"\r\n"),
            HeaderProgress::Complete(2)
        );

        let prefix = b"GET / HTTP/1.1\r\nX: ";
        let pad = MAX_HEADER - prefix.len() - 4;
        let mut exact = prefix.to_vec();
        exact.extend(std::iter::repeat_n(b'a', pad));
        exact.extend_from_slice(b"\r\n\r\n");
        assert_eq!(
            HeaderScan::new().scan(&exact),
            HeaderProgress::Complete(MAX_HEADER)
        );

        let mut oversized = prefix.to_vec();
        oversized.extend(std::iter::repeat_n(b'a', pad + 1));
        oversized.extend_from_slice(b"\r\n\r\n");
        assert_eq!(
            HeaderScan::new().scan(&oversized),
            HeaderProgress::TooLargeHeader
        );

        assert_eq!(
            HeaderScan::new().scan(&vec![b'a'; MAX_REQUEST_LINE + 1]),
            HeaderProgress::TooLargeLine
        );
    }

    #[test]
    fn head_reuses_get_head_without_body() {
        let get = dispatch_request("GET /E030 HTTP/1.1");
        let head = dispatch_request("HEAD /E030 HTTP/1.1");
        assert_eq!(head.status, "200 OK");
        assert!(head.body().is_empty());
        let rendered = head.head_parts().concat();
        assert!(
            rendered.contains(&format!("Content-Length: {}", get.body().len())),
            "HEAD must echo the GET Content-Length"
        );
    }

    #[test]
    fn date_header_is_imf_fixdate() {
        let date = http_date_header();
        assert!(date.starts_with("Date: "));
        assert!(date.ends_with(" GMT\r\n"));
    }
}
