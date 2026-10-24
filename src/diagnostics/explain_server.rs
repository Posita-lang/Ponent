use std::borrow::Cow;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use crate::diagnostics::error_code::ErrCode;

/// Serve the explanation for `code` on an already-bound `listener`.
/// The listener must have been bound to `127.0.0.1:<port>` by the caller
/// (e.g. via `TcpListener::bind("127.0.0.1:0")`).
///
/// Returns an error if no client connects within the accept budget
/// (roughly 2 s), so the caller can surface a warning instead of silently
/// pretending the page was served.  The listener's blocking mode is
/// restored before returning.
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

    listener.set_nonblocking(true)?;

    let html = build_explain_page(&ec);

    // Total accept budget: ~2 s.  A wall-clock deadline (rather than an
    // iteration count) keeps the wait bounded regardless of scheduling.
    const ACCEPT_BUDGET: Duration = Duration::from_secs(2);
    let deadline = Instant::now() + ACCEPT_BUDGET;
    let mut result: std::io::Result<()> = Err(std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        "explain server: no client connected within ~2s",
    ));
    while Instant::now() < deadline {
        match listener.accept() {
            Ok((mut stream, _)) => {
                result = serve_page(&mut stream, &html);
                break;
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => {
                result = Err(e);
                break;
            }
        }
    }
    // Browsers often open a speculative second (preconnect) connection
    // alongside the one they actually use; leaving it unaccepted gets it
    // RST on process exit, which the browser can surface as a connection
    // error.  The listener is still non-blocking here, so one extra
    // non-blocking accept is cheap insurance: drain the request if a
    // connection is already pending, otherwise skip immediately.
    if let Ok((mut extra, _)) = listener.accept() {
        let _ = drain_request(&mut extra);
        let _ = extra.shutdown(Shutdown::Both);
    }

    // The nonblocking flag must not leak into the caller's listener.  A
    // failed restore is surfaced only when the main result is Ok, so the
    // original error keeps priority.
    let restored = listener.set_nonblocking(false);
    result.and(restored)
}

/// Read and discard an HTTP request until its `\r\n\r\n` terminator, the
/// peer closes, or the wall-clock budget expires.  Shared by
/// [`serve_page`] and the speculative-connection drain in
/// [`serve_explain`].
///
/// A `\r\n\r\n` can straddle two reads, so the tail of each chunk is
/// carried into the comparison with the next chunk — a per-chunk
/// `windows(4)` scan alone misses the split and stalls a normal browser
/// request until the timeout.
fn drain_request(stream: &mut TcpStream) {
    const DRAIN_BUDGET: Duration = Duration::from_secs(1);
    let deadline = Instant::now() + DRAIN_BUDGET;
    let mut buf = [0u8; 4096];
    // Hoisted out of the loop so the large stack array is allocated once
    // instead of on every read.
    let mut head = [0u8; 3 + 4096];
    // At most a 3-byte tail carries over (`\r\n\r\n` needs 4 bytes, so 3
    // is enough to complete a split terminator).
    let mut carry = [0u8; 3];
    let mut carry_len = 0usize;
    loop {
        if Instant::now() >= deadline {
            break;
        }
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                // Prepend the carried tail from the previous read.
                head[..carry_len].copy_from_slice(&carry[..carry_len]);
                head[carry_len..carry_len + n].copy_from_slice(&buf[..n]);
                let window = &head[..carry_len + n];
                if window.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
                let take = window.len().min(3);
                carry[..take].copy_from_slice(&window[window.len() - take..]);
                carry_len = take;
            }
            Err(ref e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                // Peer has nothing more for now; stop draining.
                break;
            }
            Err(_) => break,
        }
    }
}

fn serve_page(stream: &mut TcpStream, body: &str) -> std::io::Result<()> {
    // Accepted sockets may inherit the listener's nonblocking flag on some
    // platforms; force blocking with a read timeout so draining the request
    // cannot hang the server on a silent client.  Both calls must be
    // observable: if the timeout cannot be installed, a blocking `read` could
    // wait forever, so surface the failure instead of swallowing it.
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_millis(500)))?;

    // 1) Drain the HTTP request before responding.  Closing a socket that
    //    still has unread inbound data makes the kernel send an RST instead
    //    of a FIN, which browsers may surface as ERR_CONNECTION_RESET.
    //
    //    `drain_request` bounds the drain by *wall-clock* time, not an
    //    iteration count: a byte-at-a-time client could otherwise keep
    //    `read` returning `Ok(1)` and hold the CLI for ~50 s.
    drain_request(stream);

    // 2) Respond.  Header and body are written as two chunks so the full
    //    HTML body is not copied into a combined `response` buffer.
    let header = format!(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: text/html; charset=utf-8\r\n\
         X-Content-Type-Options: nosniff\r\n\
         Content-Security-Policy: default-src 'none'; style-src 'unsafe-inline'\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n",
        body.len(),
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()?;

    // 3) Half-close our side so the peer observes a clean FIN.  The request
    //    was fully drained above, so no leftover inbound data remains to
    //    trigger a reset.
    let _ = stream.shutdown(Shutdown::Write);
    Ok(())
}

/// Escape a string for HTML embedding.
///
/// Most fields on the explain page (title, code, category, explain text)
/// come from the compiler's static error-code tables and contain no `& < >
/// " '` characters.  For those the `Cow::Borrowed` fast path avoids the
/// allocation entirely — only strings that actually need escaping are
/// owned.
fn escape_html(s: &str) -> Cow<'_, str> {
    if !s
        .bytes()
        .any(|b| matches!(b, b'&' | b'<' | b'>' | b'"' | b'\''))
    {
        return Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    Cow::Owned(out)
}

fn build_explain_page(ec: &ErrCode) -> String {
    let title = escape_html(ec.title());
    let code = escape_html(ec.code());
    let category_label = ec.category().label();
    let category = escape_html(&category_label);
    let explain = escape_html(ec.explain());
    // `url()` owns the result, so it must outlive the borrow that
    // `escape_html` makes into it.
    let url_owned = ec.url();
    let url = escape_html(&url_owned);

    // `Cow` values are used directly in `format!` — they borrow the input
    // on the clean fast path, so no intermediate `String` is allocated when
    // the fields contain no HTML-special characters.
    //
    // The colour values come from the shared `Stylesheet` (dark theme) so
    // the explain page stays in sync with the HTML renderer's palette; the
    // `:root` / `body.light` variable blocks are emitted first, then every
    // rule below references the variables instead of hard-coding hex.
    let css_vars = crate::diagnostics::emitter::stylesheet::Stylesheet::dark().to_html_css();
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{code}: {title} — Ponent</title>
<style>
{css_vars}
  body {{ font-family: 'DejaVu Sans Mono', 'Consolas', monospace; background: var(--bg); color: var(--fg); padding: 40px; max-width: 800px; margin: auto; }}
  h1 {{ color: var(--error-border); }}
  .code {{ color: var(--accent); font-weight: bold; font-size: 1.2em; }}
  .title {{ color: var(--fg); font-size: 1.1em; }}
  .category {{ color: var(--muted); margin-bottom: 20px; }}
  .explain {{ background: var(--diag-bg); padding: 20px; border-radius: 8px; line-height: 1.6; white-space: pre-wrap; color: var(--explain-fg); }}
  .url {{ color: var(--muted); margin-top: 20px; font-size: 0.85em; }}
  a {{ color: var(--accent); }}
</style>
</head>
<body>
<h1>{code}</h1>
<div class="code">{code}</div>
<div class="title">{title}</div>
<div class="category">{category}</div>
<div class="explain">{explain}</div>
<div class="url"><a href="{url}">{url}</a></div>
</body>
</html>"#,
    )
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

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
        let _ = client.read_to_end(&mut response);
        // The old code stalls for the 500 ms read timeout before draining;
        // the carry buffer must end the drain as soon as the split
        // terminator completes (~50 ms + scheduling overhead).
        assert!(
            t0.elapsed() < Duration::from_millis(500),
            "split terminator still stalled the drain for {:?}",
            t0.elapsed()
        );
        assert!(response.starts_with(b"HTTP/1.1 200 OK"));
        h.join().unwrap();
    }
}
