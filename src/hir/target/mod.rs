pub mod abi;
pub mod data_layout;
pub mod layout;
pub mod spec;

pub use abi::*;
pub use data_layout::*;
pub use spec::*;

use std::path::Path;

/// A compiled target: the user-facing spec combined with its parsed DataLayout.
#[derive(Debug, Clone)]
pub struct Target {
    pub spec: TargetSpec,
    pub data_layout: DataLayout,
}

impl Target {
    /// Look up a built-in target by name (e.g. "x86_64-linux-gnu").
    pub fn builtin(name: &str) -> Option<Self> {
        let json_str = match name {
            "x86_64-linux-gnu" | "x86_64" => include_str!("specs/x86_64_linux.json"),
            "aarch64-linux-gnu" | "aarch64" => include_str!("specs/aarch64_linux.json"),
            "riscv64" => include_str!("specs/riscv64.json"),
            "wasm32" => include_str!("specs/wasm32.json"),
            "arm-eabi" | "arm" => include_str!("specs/arm_eabi.json"),
            "custom-template" => include_str!("specs/custom_template.json"),
            _ => return None,
        };
        Self::from_json_str(json_str).or_else(|| {
            // Debug: print the actual error for diagnostics
            let _ = Self::from_json_str_debug(json_str).map_err(|e| {
                eprintln!(
                    "warning: failed to parse built-in target spec `{}`: {}",
                    name, e
                );
            });
            None
        })
    }

    /// Get the pointer width for a known architecture by looking up the
    /// built-in target specs.  Returns `None` for unknown architectures.
    /// Used by the SAT-based cfg checker to add arch→pointer_width axioms
    /// without hardcoding mappings in cfg.rs.
    /// arch -> pointer_width mapping, extracted from built-in target specs.
    const ARCH_POINTER_WIDTHS: &[(&str, u64)] = &[
        ("x86_64", 64),
        ("aarch64", 64),
        ("riscv64", 64),
        ("wasm32", 32),
        ("arm", 32),
    ];

    pub fn arch_pointer_width(arch: &str) -> Option<u64> {
        Self::ARCH_POINTER_WIDTHS
            .iter()
            .find(|(a, _)| *a == arch)
            .map(|(_, w)| *w)
    }

    /// Load a target from a JSON file path.
    pub fn from_json(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read target spec `{}`: {}", path.display(), e))?;
        // Try the debug version first to get a proper error message
        Self::from_json_str_debug(&content)
            .map_err(|e| format!("failed to parse target spec `{}`: {}", path.display(), e))
    }

    /// Parse a target from a JSON string.
    pub fn from_json_str(json: &str) -> Option<Self> {
        let spec: TargetSpec = match serde_json::from_str(json) {
            Ok(s) => s,
            Err(_) => return None,
        };
        let data_layout = match DataLayout::parse(&spec.data_layout) {
            Ok(dl) => dl,
            Err(_) => return None,
        };
        Some(Target { spec, data_layout })
    }

    /// Parse a target from a JSON string, with error details.
    /// Used for debugging target specification issues.
    pub fn from_json_str_debug(json: &str) -> Result<Self, String> {
        let spec: TargetSpec =
            serde_json::from_str(json).map_err(|e| format!("serde_json error: {}", e))?;
        // ── Validate int_sizes ───────────────────────────────────────
        for (&bits, &size) in &spec.int_sizes {
            if bits == 0 || !bits.is_power_of_two() || bits > 128 {
                return Err(format!(
                    "invalid int_sizes key: {} (must be a power of two in [1, 128])",
                    bits,
                ));
            }
            // Byte size must be at least ceil(bits/8) and at most 2× that.
            let min_size = (bits as u64 + 7) / 8;
            let max_size = min_size * 2;
            if (size as u64) < min_size || (size as u64) > max_size {
                return Err(format!(
                    "invalid int_sizes[{}]: byte size {} is out of range [{}, {}]",
                    bits, size, min_size, max_size,
                ));
            }
        }
        // ── Validate float_sizes ─────────────────────────────────────
        for (&bits, &size) in &spec.float_sizes {
            if bits != 32 && bits != 64 {
                return Err(format!(
                    "invalid float_sizes key: {} (must be 32 or 64)",
                    bits,
                ));
            }
            let expected = bits as u64 / 8;
            if (size as u64) != expected {
                return Err(format!(
                    "invalid float_sizes[{}]: byte size {} does not match expected {}",
                    bits, size, expected,
                ));
            }
        }
        let data_layout = DataLayout::parse(&spec.data_layout)
            .map_err(|e| format!("DataLayout parse error: {}", e))?;
        Ok(Target { spec, data_layout })
    }

    /// Detect the host target (the machine running the compiler).
    pub fn host() -> Self {
        // Try to detect the host from Rust's cfg attributes.
        // This is a best-effort detection; users can override with --target.
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        {
            return Self::builtin("x86_64-linux-gnu")
                .expect("built-in x86_64-linux-gnu target should be available");
        }
        #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
        {
            return Self::builtin("aarch64-linux-gnu")
                .expect("built-in aarch64-linux-gnu target should be available");
        }
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        {
            // macOS ARM — use aarch64 Linux target as fallback
            // (no macOS-specific target config yet; platform ABI is close enough
            //  for layout computation, but pointer authentication differs).
            eprintln!(
                "warning: no built-in target for macOS ARM64; using aarch64-linux-gnu as approximation"
            );
            return Self::builtin("aarch64-linux-gnu")
                .expect("built-in aarch64-linux-gnu target should be available");
        }
        #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
        {
            eprintln!(
                "warning: no built-in target for macOS x86_64; using x86_64-linux-gnu as approximation"
            );
            return Self::builtin("x86_64-linux-gnu")
                .expect("built-in x86_64-linux-gnu target should be available");
        }
        #[cfg(target_arch = "wasm32")]
        {
            return Self::builtin("wasm32").expect("built-in wasm32 target should be available");
        }
        // Unknown host — emit a warning and fall back to x86_64 Linux.
        eprintln!(
            "warning: unknown host platform (arch={}, os={}); \
             using x86_64-linux-gnu as fallback.  \
             Use `--target <spec>` to specify the correct target.",
            std::env::consts::ARCH,
            std::env::consts::OS,
        );
        Self::builtin("x86_64-linux-gnu")
            .expect("built-in x86_64-linux-gnu target should be available")
    }

    // ── Convenience accessors ────────────────────────────────────

    /// Pointer width in bits.
    pub fn pointer_width(&self) -> u8 {
        self.spec.pointer_width
    }

    /// Pointer size in bytes.
    pub fn ptr_size(&self) -> u64 {
        self.spec.pointer_width as u64 / 8
    }

    /// Pointer ABI alignment in bytes.
    pub fn ptr_align(&self) -> u64 {
        self.data_layout
            .pointer_abi_align
            .map(|(_size, abi, _pref)| abi / 8) // DataLayout stores bits; convert to bytes
            .unwrap_or(self.ptr_size())
    }

    /// Maximum alignment for any type on this target.
    pub fn max_align(&self) -> u64 {
        self.spec.max_align
    }

    /// Endianness.
    pub fn endian(&self) -> spec::Endian {
        self.spec.endian
    }

    /// Integer ABI size in bytes for a given bit width.
    pub fn int_abi_size(&self, bits: u8) -> u64 {
        if bits == 0 {
            return 0;
        }
        self.spec
            .int_sizes
            .get(&bits)
            .copied()
            .map(|s| s as u64)
            .unwrap_or_else(|| {
                // Round up to nearest byte.
                (bits as u64 + 7) / 8
            })
    }

    /// Integer ABI alignment in bytes for a given bit width.
    ///
    /// Follows Zig's rules:
    /// - `Int<0>` → alignment 1
    /// - Non-power-of-2 sizes → rounded up to the next power of 2
    /// - Architecture-specific overrides (e.g. x86 Linux: Int<64> → 4)
    pub fn int_abi_align(&self, bits: u8) -> u64 {
        if bits == 0 {
            return 1;
        }
        // Look up in data_layout integer alignments first.
        for &(b, _size, align) in &self.data_layout.integer_align {
            if b == bits {
                // DataLayout stores alignment in bits; convert to bytes.
                return align / 8;
            }
        }
        // Architecture-specific overrides (from the data_layout string).
        // For x86 (32-bit) Linux, 64-bit integers have alignment 4, not 8.
        if self.spec.arch == "x86" && self.spec.os == "linux" && bits == 64 {
            return 4;
        }
        // Fallback: compute natural alignment.
        let size = self.int_abi_size(bits);
        if size == 0 {
            return 1;
        }
        if self.spec.packed_default {
            return 1;
        }
        // Non-power-of-2 sizes: round up to next power of 2.
        let natural_align = if size.is_power_of_two() {
            size
        } else {
            size.next_power_of_two()
        };
        natural_align.min(self.max_align())
    }

    /// Float ABI size in bytes for a given bit width.
    pub fn float_abi_size(&self, bits: u8) -> u64 {
        self.spec
            .float_sizes
            .get(&bits)
            .copied()
            .map(|s| s as u64)
            .unwrap_or_else(|| bits as u64 / 8)
    }

    /// Float ABI alignment in bytes for a given bit width.
    pub fn float_abi_align(&self, bits: u8) -> u64 {
        for &(b, _size, align) in &self.data_layout.float_align {
            if b == bits {
                // align in data_layout is in bits; convert to bytes.
                return align / 8;
            }
        }
        let size = self.float_abi_size(bits);
        if self.spec.packed_default {
            1
        } else {
            size.min(self.max_align())
        }
    }
}
