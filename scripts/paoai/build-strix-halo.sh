#!/usr/bin/env bash
# Build this engine for AMD Strix Halo (gfx1151) in user space: no sudo, nothing installed system-wide.
#
#   1. ROCm SDK   = AMD TheRock 10.0.0 gfx1151 tarball (downloaded + sha256-checked, or ROCM_TARBALL=<local file>)
#   2. runtime    = pwilkin/rocm-systems @ 7dda3ac6cf (ROCr + HIP with retained-PM4 graphs), built from source
#   3. engine     = this repository (llama.cpp strix-halo @ b0f31f58 + PaoAI fix b8fe9e80), HIP backend, gfx1151
#
# The ROCr/HIP/llama.cpp cmake steps and flags are the ones pwilkin's strix-halo installer uses (install.sh @ f73872f),
# minus model downloads and launchers; this is the exact recipe the published Qwen3.8-Flash-Next numbers were measured with.
#
# usage:  scripts/paoai/build-strix-halo.sh            (run from anywhere inside this repository)
#   WORK=<dir>          where the SDK, runtime source and builds go         (default: ~/.local/share/paoai-qwen38fn)
#   ROCM_TARBALL=<file> use an already downloaded TheRock tarball          (still sha256-checked)
#   JOBS=<n>            parallel build jobs                                 (default: nproc)
# needs: git cmake ninja python3 curl tar sha256sum ldd + a C/C++ toolchain; libdrm/libelf/libnuma come from TheRock.
set -Eeuo pipefail

repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
work=${WORK:-$HOME/.local/share/paoai-qwen38fn}
jobs=${JOBS:-$(nproc)}

readonly therock_url=https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-gfx1151-10.0.0.tar.gz
readonly therock_sha256=4feabd9f2da72352df37f6d714a54847d3fe913c0341fbe2a6542c1164024baf
readonly rocm_repo_url=https://github.com/pwilkin/rocm-systems.git
readonly rocm_repo_branch=ilintar-experiments
readonly rocm_repo_commit=7dda3ac6cfe6bbe0b7f08c23a67cfa118d8641a1

rocm_root=$work/therock-10.0.0
rocm_source=$work/src/rocm-systems
rocr_build=$work/build/rocr;  rocr_install=$work/runtime/rocr
hip_build=$work/build/hip;    hip_install=$work/runtime/hip
llama_build=$work/build/engine
venv=$work/venv
log() { printf '\n=== %s  %s\n' "$(date '+%T')" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

for c in git cmake ninja python3 curl tar sha256sum ldd; do command -v "$c" >/dev/null || die "missing command: $c"; done
mkdir -p "$work/src" "$work/build" "$work/runtime" "$work/tools"

log "ROCm SDK: TheRock 10.0.0 (gfx1151)"
if [[ ! -x $rocm_root/lib/llvm/bin/clang++ ]]; then
  tarball=${ROCM_TARBALL:-$work/therock-dist-linux-gfx1151-10.0.0.tar.gz}
  [[ -f $tarball ]] || curl -fL --retry 3 -o "$tarball" "$therock_url"
  echo "$therock_sha256  $tarball" | sha256sum -c - || die "TheRock tarball sha256 mismatch"
  mkdir -p "$rocm_root"; tar xzf "$tarball" -C "$rocm_root"
fi
[[ -x $rocm_root/lib/llvm/bin/clang++ ]] || die "no clang++ in $rocm_root"
# TheRock bundles libdrm/libdrm_amdgpu/libelf/libnuma under lib/rocm_sysdeps - build against those, not system -devel packages
export PKG_CONFIG_PATH="$rocm_root/lib/rocm_sysdeps/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

# ROCr's blit kernels need `xxd -i`; some distros do not ship xxd -> tiny stand-in (only `xxd -i FILE`)
if ! command -v xxd >/dev/null; then
  cat >"$work/tools/xxd" <<'EOF'
#!/usr/bin/env python3
import re, sys
if len(sys.argv) != 3 or sys.argv[1] != "-i": sys.exit("xxd stand-in: only 'xxd -i FILE' is supported")
path = sys.argv[2]; data = open(path, "rb").read()
name = re.sub(r"[^0-9A-Za-z]", "_", path); name = "__" + name if name[:1].isdigit() else name
print(f"unsigned char {name}[] = {{")
for i in range(0, len(data), 12): print("  " + ", ".join(f"0x{b:02x}" for b in data[i:i + 12]) + ("," if i + 12 < len(data) else ""))
print(f"}};\nunsigned int {name}_len = {len(data)};")
EOF
  chmod +x "$work/tools/xxd"; export PATH="$work/tools:$PATH"
fi

log "python venv (CppHeaderParser for the HIP build)"
[[ -x $venv/bin/python ]] || python3 -m venv "$venv"
PIP_DISABLE_PIP_VERSION_CHECK=1 "$venv/bin/python" -m pip install -q 'CppHeaderParser==2.7.4'

log "runtime source: pwilkin/rocm-systems @ ${rocm_repo_commit:0:10}"
[[ -d $rocm_source/.git ]] || git clone --filter=blob:none --single-branch --branch "$rocm_repo_branch" "$rocm_repo_url" "$rocm_source"
git -C "$rocm_source" fetch -q origin "+refs/heads/$rocm_repo_branch:refs/remotes/origin/$rocm_repo_branch"
git -C "$rocm_source" merge-base --is-ancestor "$rocm_repo_commit" "origin/$rocm_repo_branch" || die "$rocm_repo_commit is no longer on $rocm_repo_branch"
git -C "$rocm_source" -c advice.detachedHead=false checkout -q --detach "$rocm_repo_commit"
[[ $(git -C "$rocm_source" rev-parse HEAD) == "$rocm_repo_commit" ]] || die "runtime not at the pinned commit"

log "ROCr"
PATH="$venv/bin:$rocm_root/bin:$PATH" cmake -S "$rocm_source/projects/rocr-runtime" -B "$rocr_build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$rocr_install" -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_PREFIX_PATH="$rocm_root;$rocm_root/lib/rocm_sysdeps" -DBUILD_SHARED_LIBS=ON
PATH="$venv/bin:$rocm_root/bin:$PATH" cmake --build "$rocr_build" --parallel "$jobs"
PATH="$venv/bin:$rocm_root/bin:$PATH" cmake --install "$rocr_build"
[[ -e $rocr_install/lib/libhsa-runtime64.so.1 ]] || die "ROCr not installed"

log "HIP (clr)"
hip_build_libs="$rocr_install/lib:$rocm_root/lib:$rocm_root/lib/llvm/lib:$rocm_root/lib/rocm_sysdeps/lib"
PATH="$venv/bin:$rocm_root/bin:$PATH" LD_LIBRARY_PATH="$hip_build_libs" cmake -S "$rocm_source/projects/clr" -B "$hip_build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$hip_install" -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_PREFIX_PATH="$rocr_install;$rocm_root;$rocm_root/lib/rocm_sysdeps" -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd \
  -DHIP_COMMON_DIR="$rocm_source/projects/hip" -DHIPCC_BIN_DIR="$rocm_root/bin" -DLLVM_ROOT="$rocm_root/lib/llvm" \
  -DClang_ROOT="$rocm_root/lib/llvm" -DROCM_PATH="$rocr_install" -Dhsa-runtime64_DIR="$rocr_install/lib/cmake/hsa-runtime64" \
  -DROCCLR_ENABLE_HSA=ON -DROCCLR_ENABLE_PAL=OFF -DHIP_ENABLE_ROCPROFILER_REGISTER=ON -DUSE_PROF_API=ON -D__HIP_ENABLE_PCH=ON
PATH="$venv/bin:$rocm_root/bin:$PATH" LD_LIBRARY_PATH="$hip_build_libs" cmake --build "$hip_build" --parallel "$jobs"
PATH="$venv/bin:$rocm_root/bin:$PATH" LD_LIBRARY_PATH="$hip_build_libs" cmake --install "$hip_build"
[[ -e $hip_install/lib/libamdhip64.so.7 ]] || die "HIP not installed"

log "engine (this repo @ $(git -C "$repo" rev-parse --short HEAD), gfx1151)"
PATH="$rocm_root/bin:$PATH" ROCM_PATH="$rocm_root" cmake -S "$repo" -B "$llama_build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$rocm_root" -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_RCCL=OFF -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=ON
PATH="$rocm_root/bin:$PATH" ROCM_PATH="$rocm_root" cmake --build "$llama_build" --parallel "$jobs" \
  --target llama-server llama-bench test-backend-ops

log "link check (the engine must load pwilkin's HIP + ROCr, not a system ROCm)"
runtime_libs="$hip_install/lib:$rocr_install/lib:$rocm_root/lib:$rocm_root/lib/llvm/lib:$rocm_root/lib/rocm_sysdeps/lib:$llama_build/bin"
out=$(LD_LIBRARY_PATH="$runtime_libs" ldd "$llama_build/bin/libggml-hip.so.0")
grep -Fq "$hip_install/lib/libamdhip64.so" <<<"$out" && grep -Fq "$rocr_install/lib/libhsa-runtime64.so" <<<"$out" \
  || { echo "$out" | grep -E 'amdhip|hsa-runtime'; die "LINK FAIL"; }
echo "LINK OK"

cat >"$work/env.sh" <<EOF
# source this before running the engine:  source $work/env.sh
export LD_LIBRARY_PATH="$runtime_libs\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export PATH="$llama_build/bin:\$PATH"
export HSA_OVERRIDE_GFX_VERSION=11.5.1        # report the GPU as gfx1151 to the runtime
export GGML_HIP_ENABLE_UNIFIED_MEMORY=1       # let the GPU use system memory (Strix Halo shares one pool)
export ENABLE_RETAINED_PM4=1                  # pwilkin runtime: retained-PM4 graph path ...
export DEBUG_HIP_GRAPH_PM4=1                  # ... and its HIP-graph switch (both need this runtime build)
EOF
log "BUILD DONE"
echo "engine:  $llama_build/bin/llama-server"
echo "next:    source $work/env.sh && llama-server --version"
