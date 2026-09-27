# paoai-qwen38fn-rocm-engine

**The engine behind the [Qwen3.8-Flash-Next PaoAI STRIX BALANCED-2.1](https://huggingface.co/PaoAI/Qwen3.8-Flash-Next-PaoAI-STRIX-BALANCED-2-GGUF) numbers on AMD Strix Halo (Ryzen AI Max+ 395 / gfx1151), ROCm/HIP.**

This is [pwilkin's llama.cpp `strix-halo` branch](https://github.com/pwilkin/llama.cpp/tree/strix-halo) at commit
`b0f31f5876ef3856b55f5bb88072cc96e5effafe`, **plus one PaoAI fix** — nothing else. It is a tested snapshot for one model
(Qwen3.8-Flash-Next), not a general-purpose llama.cpp distribution. All credit for the engine itself goes to pwilkin and the
llama.cpp authors; the full history is kept so every commit shows its real author.

| | |
|---|---|
| Base | `pwilkin/llama.cpp` branch `strix-halo` @ `b0f31f5876ef3856b55f5bb88072cc96e5effafe` (Piotr Wilkin, MIT) |
| PaoAI fix | `b8fe9e80d5b33a30415d6756a694029bc8c28738` — 1 file, 3 lines (below) |
| Runtime it needs | [`pwilkin/rocm-systems`](https://github.com/pwilkin/rocm-systems/tree/ilintar-experiments) branch `ilintar-experiments` @ `7dda3ac6cfe6bbe0b7f08c23a67cfa118d8641a1` (ROCr + HIP with retained-PM4 graphs) |
| ROCm SDK | [AMD TheRock](https://github.com/ROCm/TheRock) 10.0.0, gfx1151 tarball |
| Model | [PaoAI/Qwen3.8-Flash-Next-PaoAI-STRIX-BALANCED-2-GGUF](https://huggingface.co/PaoAI/Qwen3.8-Flash-Next-PaoAI-STRIX-BALANCED-2-GGUF) + unsloth's MTP draft sidecar |
| Measured speeds | on the model card (the only place we publish numbers) |

The original llama.cpp README is kept unchanged as [README-llama.cpp.md](README-llama.cpp.md).

## The PaoAI fix, in plain words

When the model writes, the MTP draft head guesses the next 4 tokens and the model checks all of them in one pass. On Strix Halo
(RDNA3.5), that check multiplied thin F16 weights by 4–8 columns — and at 4+ columns the engine handed the work to a hipBLAS
routine built for big matrices (128×128 tiles), which is slow for this shape. The fix lets the fast matrix-vector kernel keep the
job up to 8 columns on RDNA3.5:

```c
// ggml/src/ggml-cuda/mmvf.cu, ggml_cuda_should_use_mmvf()
if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
    return ne11 <= 8; // gfx1151: hipBLAS 128x128 tiles are ~25x slower on thin F16 weights at 4-8 columns (MTP verify)
}
```

Measured on Strix Halo: the 5-token check pass got **24–25 % faster** (8K and 32K context, 3 interleaved rounds each), 1–3 token
passes unchanged; `test-backend-ops` MUL_MAT 1297/1297 and MUL_MAT_ID 913/913 passed. Only RDNA3.5 is affected; every other
GPU takes the same path as before.

## Build (Linux, gfx1151 — the build itself runs in user space; only the one-time tool install uses sudo)

Once, install the tools (the package lists are the ones pwilkin's own installer uses for these same build steps).

Fedora:

```bash
sudo dnf install -y ca-certificates cmake curl elfutils-libelf-devel gcc gcc-c++ git libcurl-devel libdrm-devel \
  libglvnd-devel libstdc++-devel libzstd-devel make ninja-build numactl-devel openssl-devel pciutils \
  pkgconf-pkg-config python3 python3-pip python3-devel tar vim-common zlib-devel
```

Ubuntu / Debian:

```bash
sudo apt update && sudo apt install -y build-essential ca-certificates cmake curl git libcurl4-openssl-dev \
  libdrm-dev libdw-dev libelf-dev libgl-dev libnuma-dev libpciaccess-dev libssl-dev libudev-dev libzstd-dev \
  ninja-build pciutils pkg-config python3 python3-pip python3-venv xxd zlib1g-dev
```

Then, on both:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash     # the hf download tool
export PATH="$HOME/.local/bin:$PATH"               # so this terminal finds hf (new terminals do it by themselves)
```

Then build:

```bash
git clone https://github.com/guevae2/paoai-qwen38fn-rocm-engine
cd paoai-qwen38fn-rocm-engine && scripts/paoai/build-strix-halo.sh && cd ..
```

[`scripts/paoai/build-strix-halo.sh`](scripts/paoai/build-strix-halo.sh) does the three builds the measured engine used:

1. **ROCm SDK** — downloads the TheRock 10.0.0 gfx1151 tarball
   (`https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-gfx1151-10.0.0.tar.gz`, 1.79 GB) and checks its sha256
   `4feabd9f2da72352df37f6d714a54847d3fe913c0341fbe2a6542c1164024baf`. Already have it? `ROCM_TARBALL=/path/to/it`.
2. **Runtime** — clones pwilkin's `rocm-systems`, pins it to `7dda3ac6cf…`, builds ROCr and HIP (clr) into a private prefix.
3. **Engine** — builds this repository with the HIP backend for gfx1151 (`GGML_HIP=ON`, `GPU_TARGETS=gfx1151`,
   `GGML_HIP_GRAPHS=ON`, `GGML_HIP_NO_VMM=ON`, `GGML_CUDA_FA=ON`, …) and checks that it links against that private HIP + ROCr.

Everything goes to `~/.local/share/paoai-qwen38fn` (`WORK=<dir>` to change it). The step-1 packages are what a fresh system
is missing: compilers, `cmake`/`ninja`, and development headers the HIP runtime build looks for (a clean Fedora without
`libglvnd-devel` stops with `Could NOT find OpenGL`). libdrm, libelf and libnuma themselves come from the TheRock tarball.

## Run

Get the model and the draft sidecar (anonymous, no login needed):

```bash
hf download PaoAI/Qwen3.8-Flash-Next-PaoAI-STRIX-BALANCED-2-GGUF Qwen3.8-Flash-Next-PaoAI-STRIX-BALANCED-2.1.gguf --local-dir .
hf download unsloth/Qwen3.8-Flash-Next-GGUF MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf --local-dir .
mv MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf .    # hf keeps the repo's MTP/ folder; the command below expects the file here
```

Then, from the same folder:

```bash
source ~/.local/share/paoai-qwen38fn/env.sh   # written by the engine build: library path + the 4 switches explained below
llama-server -m Qwen3.8-Flash-Next-PaoAI-STRIX-BALANCED-2.1.gguf \
  -dev ROCm0 -ngl 999 -fa on -fit off --load-mode none --lazy-mode on-direct \
  -ctk f16 -ctv f16 -c 262144 -b 16384 -ub 16384 --parallel 1 --jinja \
  --spec-type draft-mtp --spec-draft-model mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  --spec-draft-device ROCm0 --spec-draft-ngl 99 --spec-draft-n-max 4 \
  --host 0.0.0.0 --port 8080
```

After about a minute the log says `listening on http://0.0.0.0:8080`; open **http://localhost:8080** for the chat page, or point an
OpenAI-compatible app at `http://localhost:8080/v1` (`--host 127.0.0.1` keeps it to this PC). Help for common problems (a missing
package, `hf: command not found`, a download that stopped halfway, GPU permission) is in the model card's "If something goes wrong"
table. These steps were run as printed on a clean Fedora 44 and Ubuntu 24.04 on 2026-09-27.

The four switches in `env.sh`: `HSA_OVERRIDE_GFX_VERSION=11.5.1` (report the GPU as gfx1151), `GGML_HIP_ENABLE_UNIFIED_MEMORY=1`
(let the GPU use system memory), `ENABLE_RETAINED_PM4=1` + `DEBUG_HIP_GRAPH_PM4=1` (pwilkin's retained-PM4 graph path — they
need his runtime build). The draft sidecar is `MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` from
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF). File checksums (sha256) are on the
[model card](https://huggingface.co/PaoAI/Qwen3.8-Flash-Next-PaoAI-STRIX-BALANCED-2-GGUF). Keep the model file on a fast NVMe
drive that is not nearly full: `--lazy-mode on-direct` reads its per-layer-embedding table from the file while writing.

## Scope and support

- **One model:** tested with Qwen3.8-Flash-Next (`qwen4exp`) on Strix Halo only. Other models and GPUs: use upstream llama.cpp
  or pwilkin's branch.
- **Frozen on purpose:** this branch stays at the tested commits so the model card's numbers stay reproducible. Newer pwilkin
  commits are not merged automatically.
- Issues about the PaoAI fix or the build script are welcome here; engine issues belong upstream.

## Credits & license

- **[pwilkin (Piotr Wilkin)](https://github.com/pwilkin/llama.cpp/tree/strix-halo)** — the Strix Halo llama.cpp branch this is,
  and the [ROCm runtime](https://github.com/pwilkin/rocm-systems/tree/ilintar-experiments) it runs on; the build steps follow
  his [strix-halo installer](https://github.com/pwilkin/strix-halo)
- **[ggml-org / llama.cpp](https://github.com/ggml-org/llama.cpp)** authors
- **[AMD ROCm TheRock](https://github.com/ROCm/TheRock)** — the ROCm 10 SDK
- **[Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)** — the MTP draft sidecar
- **[Halogen](https://github.com/peonist-ai/halogen-flash-server)** — the 8-bit dense-weights idea behind BALANCED-2.1
- **[PaoAI](https://huggingface.co/PaoAI)** — the RDNA3.5 fix, the build script, the model and its measurements

MIT License — see [LICENSE](LICENSE) (unchanged from llama.cpp). Not affiliated with AMD, ggml-org or the Qwen team.
