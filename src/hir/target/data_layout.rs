use super::spec::Endian;

/// Parsed LLVM-style data-layout string.
///
/// A data-layout string describes the target's byte order, pointer sizes,
/// integer and floating-point type alignments, and other ABI details.
///
/// Format (LLVM): `e-m:e-p:64:64:64-i64:64:64-f64:64:64-n8:16:32:64`
///
/// Components:
/// - `e` / `E` — little / big endian
/// - `m:X` — mangling mode
/// - `p:N:ABI:Pref` — pointer of size N with ABI-align and Pref-align
/// - `iN:ABI:Pref` — integer of width N with ABI-align and Pref-align
/// - `fN:ABI:Pref` — float of width N with ABI-align and Pref-align
/// - `a:ABI:Pref` — aggregate alignment
/// - `nN:N:N...` — native integer widths
/// - `vN:ABI:Pref` — vector alignment
/// - `Fi` — 128-bit float semantics
/// - `Fow` — other float semantics
#[derive(Debug, Clone)]
pub struct DataLayout {
    /// Endianness.
    pub endian: Endian,
    /// Pointer size and alignment: (size, abi_align, pref_align) in bits.
    pub pointer_abi_align: Option<(u64, u64, u64)>,
    /// Integer alignments: vec of (bits, size, abi_align).
    pub integer_align: Vec<(u8, u64, u64)>,
    /// Float alignments: vec of (bits, size, abi_align).
    pub float_align: Vec<(u8, u64, u64)>,
    /// Aggregate alignment: (abi_align, pref_align).
    pub aggregate_align: Option<(u64, u64)>,
    /// Native integer widths (e.g. [8, 16, 32, 64]).
    pub native_int_widths: Vec<u8>,
}

impl DataLayout {
    /// Parse an LLVM-style data-layout string.
    ///
    /// Returns `Err` with a description if the string is malformed.
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut endian = Endian::Little;
        let mut pointer_abi_align: Option<(u64, u64, u64)> = None;
        let mut integer_align: Vec<(u8, u64, u64)> = Vec::new();
        let mut float_align: Vec<(u8, u64, u64)> = Vec::new();
        let mut aggregate_align: Option<(u64, u64)> = None;
        let mut native_int_widths: Vec<u8> = Vec::new();

        for component in s.split('-') {
            if component.is_empty() {
                continue;
            }
            let chars: Vec<char> = component.chars().collect();
            match chars[0] {
                'e' => endian = Endian::Little,
                'E' => endian = Endian::Big,
                'p' => {
                    // p:N:ABI:Pref  (pointer)
                    // Format: `p` followed by `:` then `N:ABI:Pref`
                    let inner = if component.len() > 2 && component.as_bytes().get(1) == Some(&b':')
                    {
                        // p:N:ABI:Pref → skip "p:"
                        &component[2..]
                    } else {
                        &component[1..]
                    };
                    let parts: Vec<&str> = inner.split(':').collect();
                    if parts.len() >= 3 {
                        let size: u8 = parts[0].parse().map_err(|_| {
                            format!("invalid pointer size in data-layout: `{}`", component)
                        })?;
                        let abi: u64 = parts[1].parse().map_err(|_| {
                            format!(
                                "invalid pointer ABI alignment in data-layout: `{}`",
                                component
                            )
                        })?;
                        let pref: u64 = parts[2].parse().map_err(|_| {
                            format!(
                                "invalid pointer pref alignment in data-layout: `{}`",
                                component
                            )
                        })?;
                        pointer_abi_align = Some((size as u64, abi, pref));
                    }
                }
                'i' => {
                    // iN:ABI:Pref  (integer)
                    let inner = &component[1..];
                    let parts: Vec<&str> = inner.split(':').collect();
                    if parts.len() >= 3 {
                        let bits: u8 = parts[0].parse().map_err(|_| {
                            format!("invalid integer bit width in data-layout: `{}`", component)
                        })?;
                        let size: u64 = parts[1].parse().map_err(|_| {
                            format!("invalid integer size in data-layout: `{}`", component)
                        })?;
                        let abi: u64 = parts[2].parse().map_err(|_| {
                            format!(
                                "invalid integer ABI alignment in data-layout: `{}`",
                                component
                            )
                        })?;
                        integer_align.push((bits, size, abi));
                    }
                }
                'f' | 'F' => {
                    // fN:ABI:Pref  (float)
                    if component.starts_with("Fi") || component.starts_with("Fow") {
                        // Skip float semantics markers (Fi, Fow).
                        continue;
                    }
                    let inner = if chars[0] == 'F' {
                        &component[1..]
                    } else {
                        &component[1..]
                    };
                    let parts: Vec<&str> = inner.split(':').collect();
                    if parts.len() >= 3 {
                        let bits: u8 = parts[0].parse().map_err(|_| {
                            format!("invalid float bit width in data-layout: `{}`", component)
                        })?;
                        let size: u64 = parts[1].parse().map_err(|_| {
                            format!("invalid float size in data-layout: `{}`", component)
                        })?;
                        let abi: u64 = parts[2].parse().map_err(|_| {
                            format!(
                                "invalid float ABI alignment in data-layout: `{}`",
                                component
                            )
                        })?;
                        float_align.push((bits, size, abi));
                    }
                }
                'a' => {
                    // a:ABI:Pref  (aggregate)
                    let parts: Vec<&str> = component[1..].split(':').collect();
                    if parts.len() >= 2 {
                        let abi: u64 = parts[0].parse().map_err(|_| {
                            format!(
                                "invalid aggregate ABI alignment in data-layout: `{}`",
                                component
                            )
                        })?;
                        let pref: u64 = parts[1].parse().map_err(|_| {
                            format!(
                                "invalid aggregate pref alignment in data-layout: `{}`",
                                component
                            )
                        })?;
                        aggregate_align = Some((abi, pref));
                    }
                }
                'n' => {
                    // nN:N:N...  (native integer widths)
                    let widths: Vec<&str> = component[1..].split(':').collect();
                    for w in &widths {
                        if let Ok(bits) = w.parse::<u8>() {
                            native_int_widths.push(bits);
                        }
                    }
                }
                'm' | 'o' | 'S' | 'v' | 'w' | 'z' => {
                    // Skip components we don't need yet.
                    // m: mangling, o: object format, S: stack alignment,
                    // v: vector alignment, w: unwind table, z: default stack
                }
                _ => {
                    // Unknown component — skip.
                }
            }
        }

        Ok(DataLayout {
            endian,
            pointer_abi_align,
            integer_align,
            float_align,
            aggregate_align,
            native_int_widths,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_x86_64_linux() {
        let dl = DataLayout::parse("e-m:e-p:64:64:64-i64:64:64-f64:64:64-n8:16:32:64").unwrap();
        assert_eq!(dl.endian, Endian::Little);
        assert_eq!(dl.pointer_abi_align, Some((64, 64, 64)));
        assert!(dl.integer_align.contains(&(64, 64, 64)));
        assert!(dl.float_align.contains(&(64, 64, 64)));
        assert_eq!(dl.native_int_widths, vec![8, 16, 32, 64]);
    }

    #[test]
    fn test_parse_aarch64() {
        let dl =
            DataLayout::parse("e-m:e-p:64:64:64-i64:64:64-i128:16:128-f64:64:64-n8:16:32:64-S128")
                .unwrap();
        assert_eq!(dl.endian, Endian::Little);
        assert!(dl.integer_align.contains(&(128, 16, 128)));
        assert_eq!(dl.pointer_abi_align, Some((64, 64, 64)));
    }

    #[test]
    fn test_parse_big_endian() {
        let dl = DataLayout::parse("E-m:e-p:32:32:32-i64:64:64-n8:16:32").unwrap();
        assert_eq!(dl.endian, Endian::Big);
    }

    #[test]
    fn test_parse_empty() {
        let dl = DataLayout::parse("").unwrap();
        assert_eq!(dl.endian, Endian::Little);
        assert!(dl.pointer_abi_align.is_none());
    }
}
