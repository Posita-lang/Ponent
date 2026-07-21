use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use crate::diagnostics::error_code::ErrCode;

/// Serve the explanation for `code` on an already-bound `listener`.
/// The listener must have been bound to `127.0.0.1:<port>` by the caller
/// (e.g. via `TcpListener::bind("127.0.0.1:0")`).
pub fn serve_explain(listener: TcpListener, code: &str) -> std::io::Result<()> {
    listener.set_nonblocking(true)?;

    let ec = ErrCode::new(code);
    let html = build_explain_page(&ec);

    // Accept one connection, serve the page, then shut down.
    for _ in 0..200 {
        match listener.accept() {
            Ok((mut stream, _)) => {
                serve_page(&mut stream, &html);
                return Ok(());
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn serve_page(stream: &mut TcpStream, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {}",
        body.len(),
        body,
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

fn build_explain_page(ec: &ErrCode) -> String {
    let title = ec.title();
    let code = ec.code();
    let category = ec.category();
    let explain = ec.explain();
    let url = ec.url();

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{code}: {title} — Ponent</title>
<style>
  body {{ font-family: 'DejaVu Sans Mono', 'Consolas', monospace; background: #1e1e2e; color: #cdd6f4; padding: 40px; max-width: 800px; margin: auto; }}
  h1 {{ color: #f38ba8; }}
  .code {{ color: #89b4fa; font-weight: bold; font-size: 1.2em; }}
  .title {{ color: #cdd6f4; font-size: 1.1em; }}
  .category {{ color: #6c7086; margin-bottom: 20px; }}
  .explain {{ background: #181825; padding: 20px; border-radius: 8px; line-height: 1.6; white-space: pre-wrap; }}
  .url {{ color: #6c7086; margin-top: 20px; font-size: 0.85em; }}
  a {{ color: #89b4fa; }}
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
