#!/bin/bash
# Builds the llama.cpp libraries the packages carry into build/llama/lib: the pinned commit, as shared
# libraries with ggml's backends as modules loaded at run time (GGML_BACKEND_DL) - every x86-64 CPU
# variant, so one build runs on any machine, plus Vulkan. Only the libraries are built (libllama,
# libmtmd, ggml): no tools, no server.
#
#   scripts/build-llama.sh              build inside a container (podman or docker), so the libraries
#                                       need only that image's glibc and libstdc++ (default
#                                       ubuntu:24.04: glibc 2.39; set SEMMA_BUILD_IMAGE for another)
#   scripts/build-llama.sh --native     build with this machine's compiler (needs cmake, a C++
#                                       compiler, git, glslc, and Vulkan + SPIR-V headers)
#
# Does nothing when build/llama already holds that build. A new COMMIT ships only if
# `uv run pytest tests/test_parity.py` passes against it (same vectors, so every index is kept) and
# the ctypes structures in src/semma/embedder.py still match its headers.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMIT=8ca5a8008b57aa39f2601546c0d07912045080a5   # b12323 + 24: the build embedder.py's structures match
IMAGE=${SEMMA_BUILD_IMAGE:-docker.io/library/ubuntu:24.04}
SOURCE=build/llama.cpp
BUILD=build/llama-build
OUT=build/llama

if [[ ${1:-} != --native ]]; then
  stamp="$COMMIT $IMAGE"
  if [[ -f $OUT/lib/libllama.so && "$(cat "$OUT/BUILD" 2>/dev/null)" == "$stamp" ]]; then
    exit 0
  fi
  engine=$(command -v podman || command -v docker) || { echo "needs podman or docker (or --native)" >&2; exit 1; }
  mkdir -p build
  # glslc compiles the Vulkan backend's shaders at build time; the backend loads the system's Vulkan
  # at run time.
  "$engine" run --rm -v "$PWD:/src:Z" -w /src "$IMAGE" bash -c '
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -q > /dev/null
    apt-get install -y -q --no-install-recommends ca-certificates git cmake ninja-build g++ \
      libvulkan-dev glslc spirv-headers > /dev/null
    git config --global --add safe.directory "*"
    scripts/build-llama.sh --native
    chown -R --reference=/src build/llama build/llama.cpp build/llama-build'
  echo "$stamp" > "$OUT/BUILD"
  exit 0
fi

if [[ "$(git -C "$SOURCE" rev-parse HEAD 2>/dev/null)" != "$COMMIT" ]]; then
  rm -rf "$SOURCE"
  git init --quiet "$SOURCE"
  git -C "$SOURCE" remote add origin https://github.com/ggml-org/llama.cpp
  git -C "$SOURCE" fetch --quiet --depth 1 origin "$COMMIT"
  git -C "$SOURCE" -c advice.detachedHead=false checkout --quiet FETCH_HEAD
fi

rm -rf "$BUILD"
cmake -S "$SOURCE" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DGGML_VULKAN=ON \
  -DLLAMA_BUILD_MTMD=ON -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON '-DCMAKE_INSTALL_RPATH=$ORIGIN' > /dev/null
cmake --build "$BUILD" -j "$(nproc)" > /dev/null

# Each library once, under the name the others ask for (libllama.so.0), plus the unversioned name
# embedder.py loads; the backends (libggml-cpu-*.so, libggml-vulkan.so) are loaded by path. They find
# each other in their own folder ($ORIGIN, not the build's folders).
rm -rf "$OUT"
mkdir -p "$OUT/lib"
for file in "$BUILD"/bin/*.so*; do
  name=${file##*/}
  if [[ -L $file && $name =~ \.so\.[0-9]+$ ]]; then
    cp -L "$file" "$OUT/lib/$name"
    ln -s "$name" "$OUT/lib/${name%.*}"
  elif [[ ! -L $file && $name == *.so ]]; then
    cp "$file" "$OUT/lib/"
  fi
done
strip --strip-unneeded "$OUT"/lib/*.so.* "$OUT"/lib/libggml-*.so 2>/dev/null || true
cp "$SOURCE/LICENSE" "$OUT/LICENSE.llama.cpp"
echo "$COMMIT native" > "$OUT/BUILD"
echo "Built llama.cpp ${COMMIT:0:10} into $OUT/lib:"
ls "$OUT/lib"
