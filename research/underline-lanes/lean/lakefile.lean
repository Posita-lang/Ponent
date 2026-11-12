-- lakefile.lean
import Lake
open Lake DSL

package «underline»

@[default_target]
lean_lib «UnderlineLanes»

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "master"

