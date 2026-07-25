use serde::{Deserialize, Serialize};

/// Byte order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Endian {
    #[serde(rename = "little")]
    Little,
    #[serde(rename = "big")]
    Big,
}

/// Integer overflow policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverflowPolicy {
    #[serde(rename = "trap")]
    Trap,
    #[serde(rename = "wrap")]
    Wrap,
    #[serde(rename = "saturate")]
    Saturate,
}

/// A complete target specification, serializable to/from JSON.
///
/// Users can define custom targets for exotic hardware by writing a JSON file
/// and passing it to the compiler via `--target my_hardware.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetSpec {
    /// LLVM-style data-layout string, e.g.
    /// "e-m:e-p:64:64:64-i64:64:64-f64:64:64-n8:16:32:64"
    pub data_layout: String,

    /// Pointer width in bits (8, 16, 32, or 64).
    pub pointer_width: u8,

    /// CPU architecture name.
    pub arch: String,

    /// Operating system name.
    pub os: String,

    /// ABI name.
    pub abi: String,

    /// Byte order.
    pub endian: Endian,

    /// Integer type sizes: bit width → byte size.
    /// e.g. { "8": 1, "16": 2, "32": 4, "64": 8 }
    #[serde(default = "default_int_sizes")]
    pub int_sizes: std::collections::BTreeMap<u8, u8>,

    /// Floating-point type sizes: bit width → byte size.
    /// e.g. { "32": 4, "64": 8 }
    #[serde(default = "default_float_sizes")]
    pub float_sizes: std::collections::BTreeMap<u8, u8>,

    /// Default integer overflow policy.
    #[serde(default = "default_overflow")]
    pub default_overflow: OverflowPolicy,

    /// Maximum alignment in bytes (default 16).
    #[serde(default = "default_max_align")]
    pub max_align: u64,

    /// Whether the target supports unaligned memory access.
    #[serde(default)]
    pub allow_unaligned: bool,

    /// Whether all types default to packed layout (align=1).
    #[serde(default)]
    pub packed_default: bool,

    /// Supported atomic operation bit widths.
    #[serde(default)]
    pub atomic_bits: Option<Vec<u8>>,

    /// Stack alignment in bytes.
    #[serde(default)]
    pub stack_align: Option<u64>,

    /// Function pointer alignment in bytes.
    #[serde(default)]
    pub fn_align: Option<u64>,

    /// Global variable default alignment.
    #[serde(default)]
    pub global_align: Option<u64>,

    /// If true, panic is forbidden everywhere on this target (safety-critical).
    #[serde(default)]
    pub forbid_panic: bool,

    /// If true, all memory accesses must be deterministic (hard real-time).
    #[serde(default)]
    pub deterministic_memory: bool,

    /// Human-readable description of this target.
    #[serde(default)]
    pub description: Option<String>,
}

fn default_int_sizes() -> std::collections::BTreeMap<u8, u8> {
    let mut m = std::collections::BTreeMap::new();
    m.insert(8, 1);
    m.insert(16, 2);
    m.insert(32, 4);
    m.insert(64, 8);
    m
}

fn default_float_sizes() -> std::collections::BTreeMap<u8, u8> {
    let mut m = std::collections::BTreeMap::new();
    m.insert(32, 4);
    m.insert(64, 8);
    m
}

fn default_overflow() -> OverflowPolicy {
    OverflowPolicy::Trap
}

fn default_max_align() -> u64 {
    16
}

impl Default for TargetSpec {
    fn default() -> Self {
        TargetSpec {
            data_layout: String::new(),
            pointer_width: 64,
            arch: "unknown".into(),
            os: "freestanding".into(),
            abi: "unknown".into(),
            endian: Endian::Little,
            int_sizes: default_int_sizes(),
            float_sizes: default_float_sizes(),
            default_overflow: OverflowPolicy::Trap,
            max_align: 16,
            allow_unaligned: false,
            packed_default: false,
            atomic_bits: None,
            stack_align: None,
            fn_align: None,
            global_align: None,
            forbid_panic: false,
            deterministic_memory: false,
            description: None,
        }
    }
}
