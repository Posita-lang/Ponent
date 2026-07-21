/// Internationalisation support for diagnostic messages.
///
/// Language selection (in priority order):
///   1. `PONENT_LANG` environment variable (e.g. `en`, `ru`, `zh`)
///   2. `LANG` environment variable (system locale, e.g. `en_US.UTF-8`)
///   3. English (default)
///
/// The `tr!` macro looks up the format string key in the table for the
/// selected language, falling back to English if the key is missing.
use std::sync::atomic::{AtomicU8, Ordering};

static LANG: AtomicU8 = AtomicU8::new(0); // 0 = unset, 1 = En, 2 = Ru, 3 = Zh

/// Supported languages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    En,
    Ru,
    Zh,
}

impl Lang {
    pub fn current() -> Self {
        match LANG.load(Ordering::Relaxed) {
            0 => {
                // Detect from environment on first access.
                let lang = detect_from_env();
                LANG.store(lang as u8, Ordering::Relaxed);
                lang
            }
            1 => Lang::En,
            2 => Lang::Ru,
            3 => Lang::Zh,
            _ => Lang::En,
        }
    }

    pub fn set(lang: Lang) {
        LANG.store(lang as u8, Ordering::Relaxed);
    }
}

fn detect_from_env() -> Lang {
    let env = std::env::var("PONENT_LANG")
        .or_else(|_| std::env::var("LANG"))
        .unwrap_or_default()
        .to_lowercase();
    if env.starts_with("ru") {
        Lang::Ru
    } else if env.starts_with("zh") {
        Lang::Zh
    } else {
        Lang::En
    }
}

/// Look up a message key in the current language's table.
/// Falls back to English if the key is missing.
pub fn lookup(key: &str) -> &'static str {
    let lang = Lang::current();
    let table = match lang {
        Lang::En => EN,
        Lang::Ru => RU,
        Lang::Zh => ZH,
    };
    // Linear search — small table, not a hot path.
    for (k, v) in table.iter() {
        if *k == key {
            return v;
        }
    }
    // Fallback: try English
    for (k, v) in EN.iter() {
        if *k == key {
            return v;
        }
    }
    // Last resort: return a generic placeholder.
    // The key itself is not 'static, so we can't return it directly.
    // This path is only reached for messages that haven't been added to
    // the table yet — the key itself will be shown by the tr! macro's
    // fallback formatting.
    "(untitled)"
}

// ── Message tables ──────────────────────────────────────────────

type MsgTable = &'static [(&'static str, &'static str)];

/// English messages (default).
const EN: MsgTable = &[
    // ── General ──
    ("comptime error: {e}", "comptime error: {e}"),
    ("trait solver error: {msg}", "trait solver error: {msg}"),
    (
        "type mismatch: expected `{expected}`, found `{found}`",
        "type mismatch: expected `{expected}`, found `{found}`",
    ),
    (
        "duplicate definition of `{name}`",
        "duplicate definition of `{name}`",
    ),
    (
        "no field `{field}` found on type `{type}`",
        "no field `{field}` found on type `{type}`",
    ),
    (
        "`main` function not found in crate",
        "`main` function not found in crate",
    ),
    (
        "add a `def main() {{ ... }}` function as the entry point",
        "add a `def main() {{ ... }}` function as the entry point",
    ),
    (
        "`set` does not support pattern destructuring; use `let` instead",
        "`set` does not support pattern destructuring; use `let` instead",
    ),
    (
        "`let` requires an explicit initializer; it cannot rely on a type's default value",
        "`let` requires an explicit initializer; it cannot rely on a type's default value",
    ),
    (
        "type has no default value and no initializer provided",
        "type has no default value and no initializer provided",
    ),
    (
        "shadowing definition of `{name}`",
        "shadowing definition of `{name}`",
    ),
    (
        "impl missing method `{method}` required by trait `{trait}`",
        "impl missing method `{method}` required by trait `{trait}`",
    ),
    (
        "every trait method must be implemented — add a `def` for it in this impl block",
        "every trait method must be implemented — add a `def` for it in this impl block",
    ),
    (
        "unknown error code: `{code}`",
        "unknown error code: `{code}`",
    ),
    (
        "valid error codes include E001–E061, E101–E103, W113 — \n       run `ponent explain <CODE>` with a valid code (e.g. `ponent explain E030`)",
        "valid error codes include E001–E061, E101–E103, W113 — \n       run `ponent explain <CODE>` with a valid code (e.g. `ponent explain E030`)",
    ),
    (
        "did you mean `{candidates}`?",
        "did you mean `{candidates}`?",
    ),
    (
        "run `ponent explain <CODE>` with a valid error code",
        "run `ponent explain <CODE>` with a valid error code",
    ),
    ("Available error codes:", "Available error codes:"),
];

/// Russian messages.
const RU: MsgTable = &[
    ("comptime error: {e}", "ошибка comptime: {e}"),
    (
        "trait solver error: {msg}",
        "ошибка решателя трейтов: {msg}",
    ),
    (
        "type mismatch: expected `{expected}`, found `{found}`",
        "несоответствие типов: ожидался `{expected}`, найден `{found}`",
    ),
    (
        "duplicate definition of `{name}`",
        "повторное определение `{name}`",
    ),
    (
        "no field `{field}` found on type `{type}`",
        "поле `{field}` не найдено в типе `{type}`",
    ),
    (
        "`main` function not found in crate",
        "функция `main` не найдена в крейте",
    ),
    (
        "add a `def main() {{ ... }}` function as the entry point",
        "добавьте функцию `def main() {{ ... }}` как точку входа",
    ),
    (
        "`set` does not support pattern destructuring; use `let` instead",
        "`set` не поддерживает деструктуризацию; используйте `let`",
    ),
    (
        "`let` requires an explicit initializer; it cannot rely on a type's default value",
        "`let` требует явного инициализатора",
    ),
    (
        "type has no default value and no initializer provided",
        "тип не имеет значения по умолчанию",
    ),
    (
        "shadowing definition of `{name}`",
        "затенение определения `{name}`",
    ),
    (
        "impl missing method `{method}` required by trait `{trait}`",
        "в реализации трейта `{trait}` отсутствует метод `{method}`",
    ),
    (
        "every trait method must be implemented — add a `def` for it in this impl block",
        "все методы трейта должны быть реализованы",
    ),
    (
        "unknown error code: `{code}`",
        "неизвестный код ошибки: `{code}`",
    ),
    (
        "valid error codes include E001–E061, E101–E103, W113 — \n       run `ponent explain <CODE>` with a valid code (e.g. `ponent explain E030`)",
        "допустимые коды ошибок: E001–E061, E101–E103, W113\n       запустите `ponent explain <CODE>`",
    ),
    (
        "did you mean `{candidates}`?",
        "возможно, вы имели в виду `{candidates}`?",
    ),
    (
        "run `ponent explain <CODE>` with a valid error code",
        "запустите `ponent explain <CODE>` с правильным кодом",
    ),
    ("Available error codes:", "Доступные коды ошибок:"),
];

/// Chinese messages (fallback — used when the user explicitly sets `zh`).
const ZH: MsgTable = &[
    ("comptime error: {e}", "编译期错误：{e}"),
    ("trait solver error: {msg}", "trait 求解器错误：{msg}"),
    (
        "type mismatch: expected `{expected}`, found `{found}`",
        "类型不匹配：期望 `{expected}`，得到 `{found}`",
    ),
    ("duplicate definition of `{name}`", "重复定义 `{name}`"),
    (
        "no field `{field}` found on type `{type}`",
        "类型 `{type}` 中没有字段 `{field}`",
    ),
    (
        "`main` function not found in crate",
        "crate 中没有找到 `main` 函数",
    ),
    (
        "add a `def main() {{ ... }}` function as the entry point",
        "添加一个 `def main() {{ ... }}` 函数作为入口点",
    ),
    (
        "`set` does not support pattern destructuring; use `let` instead",
        "`set` 不支持解构，请使用 `let`",
    ),
    (
        "`let` requires an explicit initializer; it cannot rely on a type's default value",
        "`let` 需要显式初始化",
    ),
    (
        "type has no default value and no initializer provided",
        "类型没有默认值且未提供初始化器",
    ),
    ("shadowing definition of `{name}`", "`{name}` 的阴影定义"),
    (
        "impl missing method `{method}` required by trait `{trait}`",
        "trait `{trait}` 的实现缺少方法 `{method}`",
    ),
    (
        "every trait method must be implemented — add a `def` for it in this impl block",
        "所有 trait 方法都必须实现",
    ),
    ("unknown error code: `{code}`", "未知错误码：`{code}`"),
    (
        "valid error codes include E001–E061, E101–E103, W113 — \n       run `ponent explain <CODE>` with a valid code (e.g. `ponent explain E030`)",
        "有效的错误码包括 E001–E061, E101–E103, W113\n       运行 `ponent explain <CODE>` 查看详情",
    ),
    (
        "did you mean `{candidates}`?",
        "您是不是想找 `{candidates}`？",
    ),
    (
        "run `ponent explain <CODE>` with a valid error code",
        "请使用正确的错误码运行 `ponent explain <CODE>`",
    ),
    ("Available error codes:", "可用的错误码："),
];
