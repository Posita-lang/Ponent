# Chinese messages. Slugs match en.ftl exactly; never translate a slug.

comptime-error = 编译期错误：{ $e }
comptime-step-limit = 超出编译期步数限制（可能存在无限循环）
comptime-division-by-zero = 编译期表达式除以零
comptime-overflow = 编译期表达式整数溢出
comptime-type-error = 编译期类型错误：{ $e }
comptime-assertion-failed = 编译期断言失败：{ $e }
comptime-unknown-identifier = 编译期未知标识符：{ $e }
comptime-not-allowed = 编译期不允许：{ $e }
comptime-deferred = 表达式无法在编译期求值
comptime-sandbox-violation = 编译期沙箱违规：{ $e }
comptime-memory-limit = 编译期内存超限：{ $e }
comptime-internal = 编译期内部错误：{ $e }
comptime-call-stack-header = 编译期调用栈（最近调用在前）：
trait-solver-error = trait 求解器错误：{ $msg }
type-mismatch = 类型不匹配：期望 `{ $expected }`，得到 `{ $found }`
argument-type-mismatch = 调用 `{ $callee }` 时实参类型不匹配：参数 `{ $param }` 期望 `{ $expected }`，得到 `{ $found }`
contract-non-bool = `{ $clause }` 子句必须为布尔值，得到 `{ $found }`
impl-missing-method = `{ $trait }` 的 impl 缺少方法 `{ $method }`
duplicate-definition = 重复定义 `{ $name }`
field-not-found = 类型 `{ $type }` 中没有字段 `{ $field }`
name-not-found = 未找到名称：`{ $name }`
main-not-found = crate 中没有找到 `main` 函数
set-no-destructuring = `set` 不支持解构，请使用 `let`
let-requires-initializer = `let` 需要显式初始化
no-default-value = 类型没有默认值且未提供初始化器
shadowing-definition = `{ $name }` 的阴影定义
missing-method = trait `{ $trait }` 的实现缺少方法 `{ $method }`
impl-trait-methods = 所有 trait 方法都必须实现
unknown-error-code = 未知错误码：`{ $code }`
did-you-mean = 您是不是想找 `{ $candidates }`？
run-explain-hint = 请使用正确的错误码运行 `ponent explain <CODE>`
available-error-codes = 可用的错误码：

summary-errors = 因之前的 { $count } 个错误中止
summary-mixed-errors = 因之前的 { $count } 个错误中止
summary-warnings = 已发出 { $count } 个警告
summary-files = 涉及 { $count } 个文件

# 诊断级别标签 (P3-23: i18n for DiagnosticLevel::label)
level-fatal = 致命错误
level-error = 错误
level-warning = 警告
level-help = 帮助
level-note = 注释
level-info = 信息

# 调用链条目类型 (P3-23: i18n for ChainEntryKind::as_str)
chain-call = 在此处调用
chain-inlined = 在此处内联
chain-reference = 在此处引用
chain-defines = 在此处定义

# 编译期求值原因 (P3-23: i18n for ComptimeReason::as_str)
comptime-reason-block = comptime {"{"} ... {"}"} 块
comptime-reason-fndef = comptime def 函数体
comptime-reason-fncall = comptime 函数调用
comptime-reason-test = @comptime_test 函数
comptime-reason-assert = assert!()
comptime-reason-typeinfo = @typeInfo!()
comptime-reason-layoutof = layout_of!()

# 错误类别 (P3-23: i18n for ErrorCategory::label)
category-parse = 解析错误
category-resolution = 解析错误
category-type = 类型错误
category-contract = 契约错误
category-trait = Trait 错误
category-inference = 类型推导错误
category-internal = 内部错误
category-generic = 错误

# Humanizer 标签 (P3-24: i18n for DiagnosticKind::labels)
label-expected = 期望 { $expected }
label-field-not-found = 字段未找到
label-expected-found = 期望 `{ $expected }`，得到 `{ $found }`
label-declared-here = `{ $expected }` 在此处声明
label-not-in-scope = 在此作用域中未找到
label-duplicate-definition = 重复定义
label-first-defined-here = 首次定义于此
label-expected-bool = 期望 bool
label-method-missing = 方法缺失于此
label-required-by-trait = trait 声明要求于此
label-comptime-eval-error = comptime 求值错误

# Humanizer 帮助 (P3-24: i18n for DiagnosticKind::help)
help-cast-or-change = 尝试使用 `as` 进行转换，或更改表达式类型

# 错误码解释/建议 (P3-23: i18n for explain_error_code/suggest_code)
help-valid-codes = 有效的错误码包括 { $codes }
run-explain-hint-detail = 请使用有效的错误码运行 `ponent explain <CODE>`
run-explain-hint-example = 请使用有效的错误码运行 `ponent explain <CODE>`（例如 `ponent explain E030`）
run-explain-detail = 运行 `ponent explain <CODE>` 以获取特定错误码的详细信息。
unknown-error = 未知错误

# Batch trailers / fold marker / call-chain header (renderers)
suppressed-note = 抑制了 { $count } 个额外错误
fold-marker-lines = 省略了 { $gap } 行
chain-header = 被引用于:
html-doc-title = Ponent 编译器诊断
html-h1 = Ponent 编译器诊断
html-toggle-theme = 切换主题
html-summary = { $errors } 个错误，{ $warnings } 个警告
html-summary-files = 涉及 { $files } 个文件
type-ctx-annotation = 来自类型标注
type-ctx-return-type = 来自返回类型
type-ctx-binop = 来自运算符
type-ctx-function-arg = 来自函数参数
type-ctx-field = 来自字段类型
type-ctx-type-alias = 来自类型别名
type-ctx-contract = 来自契约
type-ctx-inference = 来自类型推断
