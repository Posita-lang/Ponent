use std::borrow::Cow;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use crate::diagnostics::error_code::ErrCode;
use crate::smallvec::SmallVec;
use compact_str::CompactString;

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
        drain_request(&mut extra, &[]);
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
fn drain_request(stream: &mut TcpStream, prefix: &[u8]) {
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

    // Bytes the request-line reader already pulled past the `\n` may already
    // contain the `\r\n\r\n` terminator (or a split piece of it), so check
    // them before touching the stream.  Only the trailing 3 bytes can
    // participate in a split terminator, so carry those into the loop —
    // matching the inter-read carry below.
    if !prefix.is_empty() {
        if prefix.windows(4).any(|w| w == b"\r\n\r\n") {
            return;
        }
        let take = prefix.len().min(3);
        carry[..take].copy_from_slice(&prefix[prefix.len() - take..]);
        carry_len = take;
    }

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
    drain_request(stream, &[]);

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

    // Curated or category-sibling related codes for a "see also"
    // section — the same source of truth used by the terminal renderer.
    let explicit = crate::diagnostics::error_code::related_codes_of(ec.code());
    let related: SmallVec<[&'static str; 4]> = if explicit.is_empty() {
        crate::diagnostics::error_code::category_siblings_of(ec.code(), 4)
    } else {
        SmallVec::from_slice(explicit)
    };
    let see_also = if related.is_empty() {
        String::new()
    } else {
        let mut s = String::from("<div class=\"see-also\"><h3>See also</h3><ul>\n");
        for rel in &related {
            let title = crate::diagnostics::error_code::lookup(rel)
                .map(|e| e.title)
                .unwrap_or("(unknown)");
            let _ = std::fmt::Write::write_fmt(
                &mut s,
                format_args!(
                    "<li><a href=\"/{code}\"><span class=\"code\">{code}</span></a> — {title}</li>\n",
                    code = escape_html(rel),
                    title = escape_html(title),
                ),
            );
        }
        s.push_str("</ul></div>\n");
        s
    };

    // Dark theme is the `:root` default; `body.light` overrides it.  The
    // toggle below flips the class and persists the choice in localStorage,
    // matching `HtmlRenderer`'s theme toggle.
    let css_vars = crate::diagnostics::emitter::stylesheet::dark_css_cached();
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{code}: {title} — Ponent</title>
<style>
{css_vars}
  body {{ font-family: 'DejaVu Sans Mono', 'Consolas', monospace; background: var(--bg); color: var(--fg); padding: 40px; max-width: 800px; margin: auto; }}
  #theme-toggle {{ position: fixed; top: 16px; right: 16px; background: var(--bg); color: var(--fg); border: 1px solid var(--fg); border-radius: 6px; padding: 6px 14px; cursor: pointer; font-family: inherit; font-size: 0.85em; }}
  #theme-toggle:hover {{ opacity: 0.8; }}
  h1 {{ color: var(--error-border); }}
  .code {{ color: var(--accent); font-weight: bold; font-size: 1.2em; }}
  .title {{ color: var(--fg); font-size: 1.1em; }}
  .category {{ color: var(--muted); margin-bottom: 20px; }}
  .explain {{ background: var(--diag-bg); padding: 20px; border-radius: 8px; line-height: 1.6; white-space: pre-wrap; color: var(--explain-fg); }}
  .url {{ color: var(--muted); margin-top: 20px; font-size: 0.85em; }}
  a {{ color: var(--accent); }}
  .see-also {{ margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--fg); }}
  .see-also h3 {{ color: var(--accent); font-size: 1em; margin: 0 0 8px 0; }}
  .see-also ul {{ list-style: none; padding: 0; margin: 0; }}
  .see-also li {{ margin: 4px 0; }}
  .see-also .code {{ font-weight: bold; }}
</style>
<script>
(function(){{
  var t = localStorage.getItem('ponent-theme');
  if (t === 'light' || (!t && matchMedia('(prefers-color-scheme: light)').matches)) {{
    document.body.classList.add('light');
  }}
}})();
</script>
</head>
<body>
<button id="theme-toggle" type="button">Toggle theme</button>
<h1>{code}</h1>
<div class="code">{code}</div>
<div class="title">{title}</div>
<div class="category">{category}</div>
<div class="explain">{explain}</div>
{see_also}<div class="url"><a href="{url}">{url}</a></div>
<script>
(function(){{
  document.getElementById('theme-toggle').addEventListener('click', function(){{
    var light = document.body.classList.toggle('light');
    localStorage.setItem('ponent-theme', light ? 'light' : 'dark');
  }});
}})();
</script>
</body>
</html>"#,
    )
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
/// The HTTP handling is deliberately minimal: request line only, no
/// keep-alive, no chunked transfer, one connection per request.
pub fn serve_explain_persistent(listener: &TcpListener) -> std::io::Result<()> {
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
    for incoming in listener.incoming() {
        match incoming {
            Ok(mut stream) => {
                // Per-connection errors must not kill the server.
                let _ = handle_persistent_connection(&mut stream);
            }
            // EAGAIN/EINTR on some platforms — retry.
            Err(_) => continue,
        }
    }
    Ok(())
}

fn handle_persistent_connection(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_millis(500)))?;
    let request_line = read_request_line(stream);
    let (line, leftover) = match request_line {
        Some(pair) => pair,
        None => return Ok(()),
    };
    // Drain the rest of the request before responding; closing a socket
    // with unread inbound data makes the kernel send RST instead of FIN.
    // `leftover` is the bytes `read_request_line` already consumed past
    // the request `\n` — hand them to `drain_request` so they are not lost
    // (the next `read` would otherwise miss them).
    drain_request(stream, &leftover);
    let body = dispatch_request(&line);
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
    let _ = stream.shutdown(Shutdown::Write);
    Ok(())
}

/// Read one line (up to and including the `\r\n`) from the stream, plus any
/// bytes the same read pulled past the `\n`.
///
/// Returns `(line, leftover)`: `line` is the request line without the
/// trailing CRLF, and `leftover` is everything after the `\n` that arrived
/// in the same `read` chunk.  Those bytes belong to the request headers and
/// must be handed to [`drain_request`] as its `prefix` — otherwise the next
/// `read` would skip them and the drain could miss the `\r\n\r\n` terminator.
///
/// Reads in 512-byte chunks (one `read` syscall per chunk) instead of one
/// byte at a time, so a typical ~20–80 byte request line is one syscall.
/// `line` is a `SmallVec<[u8; 512]>` so it stays on the stack for the common
/// request-line length, and `leftover` is bounded by the chunk size so its
/// `SmallVec<[u8; 512]>` is inline too.  The returned `CompactString` is
/// built from the validated `&str`, so a short request line also inlines
/// (≤24 bytes) and the common path makes zero heap allocations.
fn read_request_line(stream: &mut TcpStream) -> Option<(CompactString, SmallVec<[u8; 512]>)> {
    let mut line: SmallVec<[u8; 512]> = SmallVec::new();
    let mut buf = [0u8; 512];
    loop {
        let n = match stream.read(&mut buf) {
            Ok(0) => return None,
            Ok(n) => n,
            Err(_) => return None,
        };
        let chunk = &buf[..n];
        match memchr::memchr(b'\n', chunk) {
            Some(nl) => {
                if line.len() + nl > 8192 {
                    return None;
                }
                line.extend_from_slice(&chunk[..nl]);
                if line.ends_with(b"\r") {
                    line.pop();
                }
                let mut leftover: SmallVec<[u8; 512]> = SmallVec::new();
                leftover.extend_from_slice(&chunk[nl + 1..]);
                let line = std::str::from_utf8(&line[..]).ok()?.into();
                return Some((line, leftover));
            }
            None => {
                if line.len() + n > 8192 {
                    return None;
                }
                line.extend_from_slice(chunk);
            }
        }
    }
}

/// Serve a cached explain page for `ec`.
///
/// `build_explain_page` is a pure function of the static error-code tables,
/// so each code's page is built at most once and reused for the lifetime of
/// the persistent server.  The first request for a code pays one build
/// (escaping + sibling lookup + `format!`); every later request is an
/// `Arc<str>` refcount bump instead of a full `String` clone.  The read
/// guard is dropped before building so two concurrent requests for different
/// codes do not serialize their builds.
fn explain_page_cached(ec: &ErrCode) -> std::sync::Arc<str> {
    static PAGES: std::sync::OnceLock<
        std::sync::Mutex<rustc_hash::FxHashMap<CompactString, std::sync::Arc<str>>>,
    > = std::sync::OnceLock::new();
    let pages = PAGES.get_or_init(|| std::sync::Mutex::new(rustc_hash::FxHashMap::default()));
    let key = CompactString::from(ec.code());
    {
        let guard = pages.lock().unwrap();
        if let Some(page) = guard.get(&key) {
            return std::sync::Arc::clone(page);
        }
    }
    let page: std::sync::Arc<str> = std::sync::Arc::from(build_explain_page(ec));
    pages
        .lock()
        .unwrap()
        .insert(key, std::sync::Arc::clone(&page));
    page
}

fn dispatch_request(request_line: &str) -> std::sync::Arc<str> {
    // `GET /E030 HTTP/1.1` → path `/E030`
    let path = request_line.split_whitespace().nth(1).unwrap_or("/");
    // Strip query string.
    let path = path.split('?').next().unwrap_or("/");
    let decoded = percent_decode(path);
    if decoded == "/" || decoded.is_empty() {
        return build_index_page();
    }
    let code = decoded.trim_start_matches('/');
    match ErrCode::try_new(code) {
        Ok(ec) => explain_page_cached(&ec),
        Err(_) => std::sync::Arc::from(build_not_found_page(code)),
    }
}

/// Percent-decode `%XX` sequences.  Browsers occasionally send encoded
/// paths; without decoding, `/E030%0A` would miss the registry lookup.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(hex) = std::str::from_utf8(&bytes[i + 1..i + 3])
            && let Ok(b) = u8::from_str_radix(hex, 16)
        {
            out.push(b);
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
        Ok(s) => s,
        Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
    }
}

fn build_index_page() -> std::sync::Arc<str> {
    // The index is a pure function of the static CODE_TABLE, so build it
    // once and bump the refcount per request — cloning one ~10 KB `Arc<str>`
    // is far cheaper than re-escaping 452 codes and re-formatting every
    // entry on each `GET /`.
    static INDEX: std::sync::OnceLock<std::sync::Arc<str>> = std::sync::OnceLock::new();
    std::sync::Arc::clone(INDEX.get_or_init(|| std::sync::Arc::from(build_index_page_uncached())))
}

fn build_index_page_uncached() -> String {
    let mut body = String::with_capacity(8 * 1024);
    body.push_str(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ponent error codes</title>
<style>
body { font-family: 'DejaVu Sans Mono', 'Consolas', monospace; background: #1e1e2e; color: #cdd6f4; padding: 40px; max-width: 900px; margin: auto; }
h1 { color: #f38ba8; }
h2 { color: #89b4fa; margin-top: 28px; }
ul { list-style: none; padding: 0; }
li { margin: 4px 0; }
a { color: #89b4fa; text-decoration: none; }
a:hover { text-decoration: underline; }
.code { font-weight: bold; }
.title { color: #bac2de; }
</style>
</head>
<body>
<h1>Ponent error codes</h1>
"#,
    );
    for (cat, entries) in crate::diagnostics::error_code::code_entries_by_category() {
        body.push_str(&format!("<h2>{}</h2>\n<ul>\n", escape_html(&cat.label())));
        for entry in entries {
            let code = escape_html(entry.code);
            let title = escape_html(entry.title);
            body.push_str(&format!(
                "<li><a href=\"/{code}\"><span class=\"code\">{code}</span></a> — \
                 <span class=\"title\">{title}</span></li>\n",
            ));
        }
        body.push_str("</ul>\n");
    }
    body.push_str("</body>\n</html>\n");
    body
}

fn build_not_found_page(code: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Not found — Ponent</title>
<style>
body {{ font-family: monospace; background: #1e1e2e; color: #cdd6f4; padding: 40px; }}
h1 {{ color: #f38ba8; }}
a {{ color: #89b4fa; }}
</style>
</head>
<body>
<h1>404 — not found</h1>
<p>No error code <code>{}</code> is registered.</p>
<p><a href="/">Back to the index</a></p>
</body>
</html>"#,
        escape_html(code),
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
