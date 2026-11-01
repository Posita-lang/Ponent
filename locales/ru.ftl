# Russian messages. Slugs match en.ftl exactly; never translate a slug.

comptime-error = ошибка comptime: { $e }
trait-solver-error = ошибка решателя трейтов: { $msg }
type-mismatch = несоответствие типов: ожидался `{ $expected }`, найден `{ $found }`
argument-type-mismatch = несоответствие типов аргумента в вызове `{ $callee }`: параметр `{ $param }` ожидал `{ $expected }`, найден `{ $found }`
contract-non-bool = условие `{ $clause }` должно быть логическим значением, найдено `{ $found }`
impl-missing-method = в impl `{ $trait }` отсутствует метод `{ $method }`
duplicate-definition = повторное определение `{ $name }`
field-not-found = поле `{ $field }` не найдено в типе `{ $type }`
main-not-found = функция `main` не найдена в крейте
set-no-destructuring = `set` не поддерживает деструктуризацию; используйте `let`
let-requires-initializer = `let` требует явного инициализатора
no-default-value = тип не имеет значения по умолчанию
shadowing-definition = затенение определения `{ $name }`
missing-method = в реализации трейта `{ $trait }` отсутствует метод `{ $method }`
impl-trait-methods = все методы трейта должны быть реализованы
unknown-error-code = неизвестный код ошибки: `{ $code }`
did-you-mean = возможно, вы имели в виду `{ $candidates }`?
run-explain-hint = запустите `ponent explain <CODE>` с правильным кодом
available-error-codes = Доступные коды ошибок:

summary-errors =
    { $count ->
        [one]   прерывание из-за предыдущей ошибки
        [few]   прерывание из-за { $count } предыдущих ошибок
       *[other] прерывание из-за { $count } предыдущих ошибок
    }

summary-mixed-errors =
    { $count ->
        [one]   прерывание из-за { $count } предыдущей ошибки
        [few]   прерывание из-за { $count } предыдущих ошибок
       *[other] прерывание из-за { $count } предыдущих ошибок
    }

summary-warnings =
    { $count ->
        [one]   1 предупреждение выпущено
        [few]   { $count } предупреждения выпущено
       *[other] { $count } предупреждений выпущено
    }

summary-files =
    { $count ->
        [one]   в 1 файле
        [few]   в { $count } файлах
       *[other] в { $count } файлах
    }

# Метки уровней диагностики (P3-23: i18n для DiagnosticLevel::label)
level-fatal = фатальная ошибка
level-error = ошибка
level-warning = предупреждение
level-help = подсказка
level-note = заметка
level-info = инфо

# Виды записей цепочки вызовов (P3-23: i18n для ChainEntryKind::as_str)
chain-call = вызов из
chain-inlined = инлайн из
chain-reference = ссылка из
chain-defines = определено в

# Причины comptime-вычисления (P3-23: i18n для ComptimeReason::as_str)
comptime-reason-block = comptime {"{"} ... {"}"} блок
comptime-reason-fndef = тело функции comptime def
comptime-reason-fncall = вызов comptime-функции
comptime-reason-test = функция @comptime_test
comptime-reason-assert = assert!()
comptime-reason-typeinfo = @typeInfo!()
comptime-reason-layoutof = layout_of!()

# Категории ошибок (P3-23: i18n для ErrorCategory::label)
category-parse = Ошибка разбора
category-resolution = Ошибка разрешения
category-type = Ошибка типов
category-contract = Ошибка контракта
category-trait = Ошибка трейта
category-inference = Ошибка вывода типов
category-internal = Внутренняя ошибка
category-generic = Ошибка

# Метки Humanizer (P3-24: i18n для DiagnosticKind::labels)
label-expected = ожидалось { $expected }
label-field-not-found = поле не найдено
label-expected-found = ожидалось `{ $expected }`, найдено `{ $found }`
label-declared-here = `{ $expected }` объявлено здесь
label-not-in-scope = не найдено в этой области
label-duplicate-definition = повторное определение
label-first-defined-here = впервые определено здесь
label-expected-bool = ожидался bool
label-method-missing = метод отсутствует здесь
label-required-by-trait = требуется объявлением трейта здесь
label-comptime-eval-error = ошибка comptime-вычисления

# Подсказки Humanizer (P3-24: i18n для DiagnosticKind::help)
help-cast-or-change = попробуйте использовать `as` для приведения или измените тип выражения

# Объяснение кода ошибки / подсказки (P3-23: i18n для explain_error_code/suggest_code)
help-valid-codes = допустимые коды ошибок включают { $codes }
run-explain-hint-detail = запустите `ponent explain <CODE>` с допустимым кодом ошибки
run-explain-hint-example = запустите `ponent explain <CODE>` с допустимым кодом (например `ponent explain E030`)
run-explain-detail = Запустите `ponent explain <CODE>` для подробностей о конкретном коде ошибки.
unknown-error = неизвестная ошибка
