#!/usr/bin/env python3
"""Build the bundled THIRD_PARTY_NOTICES.txt from the licences of what is shipped.

Usage: scripts/notices.py <staged-python-prefix> <out.txt>

The prefix is build/stage/usr/lib/semma/python in scripts/package.sh: the bundled
CPython with every wheel installed, *before* the PySide6/Qt trim. Only the Python
3 standard library is used. Sources that are not in the wheels (Qt, PySide,
python-build-standalone component licences, ICU) are downloaded once into
build/notices-src/ and cached there. Every required file is checked: a missing
licence fails the build instead of producing an incomplete notice.
"""

import fnmatch
import os
import re
import subprocess
import sys
from email.parser import Parser
from pathlib import Path

RULE = "=" * 78

PREAMBLE = """\
Semma — third-party software notices

Semma's own code is MIT licensed (see LICENSE). Dependencies keep their own terms.
The model, EmbeddingGemma 2 by Google DeepMind, is Apache-2.0 licensed and is
downloaded separately from https://huggingface.co/ggml-org/embeddinggemma-2-GGUF.

Qt/PySide6 and shiboken6 {qt_version} use the LGPL-3.0 option; ICU is {icu_version}.
Sources: https://download.qt.io/official_releases/qt/{qt_maj_min}/{qt_version}/submodules/
https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-{qt_version}-src/
https://github.com/unicode-org/icu/tree/release-{icu_major}-{icu_minor}
Qt libraries and plugins remain shared libraries, replaceable with ABI-compatible
versions in /usr/lib/semma/python/lib/python{py_short}/site-packages/PySide6/Qt/.
PySide6 and shiboken6 can likewise be replaced in that site-packages directory.
Semma places no restrictions on modifying or reverse-engineering these libraries.

Python is CPython {py_version}, python-build-standalone release {pbs_release}.
Sources and upstream build recipes/patches:
https://www.python.org/ftp/python/{py_version}/Python-{py_version}.tar.xz
https://github.com/astral-sh/python-build-standalone/tree/{pbs_release}
https://github.com/astral-sh/python-build-standalone/blob/{pbs_release}/pythonbuild/downloads.json
Its Berkeley DB 6.0.19 backend has Sleepycat source-availability terms:
https://ftp.osuosl.org/pub/blfs/conglomeration/db/db-6.0.19.tar.gz
Semma source and packaging: https://github.com/Mihuy1/semma
Semma trims unused Python/Qt components, not their source code.

The texts below are collected by scripts/notices.py at build time from the files
the package ships and from version-matched upstream sources. Upstream notice sets
may mention optional components removed from Semma's package. Shared texts are
printed once, after the names of the components that ship them.
"""

LICENCE_FILE = re.compile(r"^(licen[cs]e|copying|notice|copyright|unlicense)", re.I)
# Extract only licence files from the upstream source archives, never the tree.
TAR_PATTERNS = (
    "*/LICENSES/*",
    "*LICENSE*",
    "*LICENCE*",
    "*COPYING*",
    "*[Cc]opyright*",
    "*NOTICE*",
    "*/qt_attribution.json",
    # The shiboken support header carries a PSF licence in its body.
    "*/bufferprocs_py37.h",
    "*/PSF-*.txt",
)

# Licence files of Qt's tests, examples and docs: none of that is in the package.
SKIP_PATTERNS = ("*/tests/*", "*/examples/*", "*/doc/*")

# Wheels that ship no licence file, only metadata: their texts from the release's source tag.
# {v} is the installed version. A new wheel without licence files fails the build until listed.
UPSTREAM_LICENCES = {
    "sqlite-vec": ("https://github.com/asg017/sqlite-vec", ["LICENSE-MIT", "LICENSE-APACHE"]),
}

out: list[str] = []


def need(path: Path | str) -> Path:
    """A licence we must print is missing: fail the build with a clear message."""
    path = Path(path)
    if not path.is_file():
        sys.exit(f"notices: required file is missing: {path}")
    return path


def read(path: Path | str) -> str:
    return need(path).read_text(encoding="utf-8", errors="replace")


def need_dir(path: Path | str) -> Path:
    path = Path(path)
    if not path.is_dir():
        sys.exit(f"notices: required directory is missing: {path}")
    return path


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd, check=True, text=True, capture_output=capture,
        )
    except FileNotFoundError:
        sys.exit(f"notices: command not found: {cmd[0]}")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"notices: command failed ({' '.join(cmd)}): {exc}")


def section(title: str, *texts: tuple[str, str]) -> None:
    """Add a section to the output.

    Each text is (source, content). Sources with whitespace-identical content are
    printed once: their names are listed in the heading, above the shared text.
    """
    groups: dict[str, tuple[str, list[str]]] = {}
    for source, content in texts:
        content = content.strip("\n")
        key = " ".join(content.split())
        if key in groups:
            groups[key][1].append(source)
        else:
            groups[key] = (content, [source])
    for content, sources in groups.values():
        heading = (
            sources[0] if len(sources) == 1
            else f"{title}: {', '.join(sources)}"
        )
        out.extend((RULE, heading, RULE, ""))
        out.extend(content.splitlines())
        out.append("")


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def dist_metadata(dist: Path) -> dict:
    msg = Parser().parsestr(read(dist / "METADATA"))
    classifiers = msg.get_all("Classifier") or []
    license = (
        msg.get("License-Expression")
        or msg.get("License")
        or next(
            (c.split("::")[-1].strip() for c in classifiers
             if c.lower().startswith("license")),
            None,
        )
    )
    if not license or license.strip().upper() in {"UNKNOWN", "NONE"}:
        license = None
    home = msg.get("Home-page")
    project_urls = msg.get_all("Project-URL") or []
    source = next(
        (u.partition(",")[2].strip() for u in project_urls
         if u.partition(",")[0].strip().lower()
         in {"source", "homepage", "home", "repository", "code"}),
        None,
    ) or (project_urls[0].partition(",")[2].strip() if project_urls else home)
    return {
        "name": msg.get("Name") or dist.name.split("-")[0],
        "version": msg.get("Version") or "0",
        "license": license,
        "source": source or "(not given in metadata)",
    }


def version_from_binary(path: Path, major: str) -> str:
    """libicuuc.so.73 in the wheel has no minor in its name; ICU stores it in the .so."""
    data = path.read_bytes()
    found = set(
        m.group(1).decode("ascii")
        for m in re.finditer(
            rb"(?<![0-9])" + major.encode() + rb"\.(\d{1,3})(?![0-9])", data
        )
    )
    if len(found) != 1:
        sys.exit(
            f"notices: cannot determine the ICU minor version from {path} "
            f"(found {sorted(found)})"
        )
    return f"{major}.{found.pop()}"


def icu_version(qt_lib: Path) -> str:
    libs = sorted(qt_lib.glob("libicuuc.so.*"))
    if not libs:
        sys.exit(f"notices: no libicuuc.so.* in {qt_lib}; cannot determine ICU version")
    lib = libs[0]
    match = re.search(r"\.so\.(\d+)\.(\d+)", lib.name)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    match = re.search(r"\.so\.(\d+)$", lib.name)
    if not match:
        sys.exit(f"notices: cannot parse ICU version from {lib.name}")
    return version_from_binary(lib, match.group(1))


def fetch(url: str, dest: Path) -> Path:
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["curl", "-fsSI", "-o", os.devnull, url])
    part = dest.with_name(dest.name + ".part")
    run(["curl", "-fsSL", "-o", str(part), url])
    os.replace(part, dest)
    return dest


def extract(archive: Path, args: list[str], dest: Path) -> Path:
    """Unpack into dest.part, then rename: a failed run is never mistaken for a cached one. The
    archive is deleted afterwards (only its licence files are kept)."""
    if dest.is_dir():
        return dest
    part = dest.with_name(dest.name + ".part")
    subprocess.run(["rm", "-rf", str(part)], check=True)
    part.mkdir(parents=True)
    run(["tar", "-xf", str(archive), "-C", str(part), *args])
    if not any(part.rglob("*")):
        sys.exit(f"notices: nothing was extracted from {archive}")
    os.replace(part, dest)
    archive.unlink()
    return dest


def source_licences(cache: Path, qt_version: str, py_version: str, pbs: str) -> tuple[Path, Path, Path, Path]:
    """Fetch and extract the licence files missing from the wheels; cache them."""
    qt_base_url = (
        f"https://download.qt.io/official_releases/qt/"
        f"{'.'.join(qt_version.split('.')[:2])}/{qt_version}/submodules/"
    )
    pyside_url = (
        "https://download.qt.io/official_releases/QtForPython/pyside6/"
        f"PySide6-{qt_version}-src/"
    )
    pbs_asset = (
        f"cpython-{py_version}+{pbs}-x86_64-unknown-linux-gnu-pgo+lto-full.tar.zst"
    )
    pbs_url = (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{pbs}/{pbs_asset}"
    )
    urls = {
        "qtbase": (
            qt_base_url + f"qtbase-everywhere-src-{qt_version}.tar.xz",
            cache / f"qtbase-everywhere-src-{qt_version}.tar.xz",
            cache / f"qtbase-{qt_version}",
        ),
        "qtwayland": (
            qt_base_url + f"qtwayland-everywhere-src-{qt_version}.tar.xz",
            cache / f"qtwayland-everywhere-src-{qt_version}.tar.xz",
            cache / f"qtwayland-{qt_version}",
        ),
        "pyside": (
            pyside_url + f"pyside-setup-everywhere-src-{qt_version}.tar.xz",
            cache / f"pyside-setup-everywhere-src-{qt_version}.tar.xz",
            cache / f"pyside-{qt_version}",
        ),
    }
    dirs = {}
    for key, (url, archive, dest) in urls.items():
        if not dest.is_dir():
            fetch(url, archive)
            members = run(["tar", "-tf", str(archive)], capture=True).stdout.splitlines()
            patterns = [
                p for p in TAR_PATTERNS
                if any(fnmatch.fnmatch(m, p) for m in members)
            ]
            if not patterns:
                sys.exit(f"notices: no licence files found in {archive}")
            extract(archive, ["--strip-components=1", "--wildcards", *patterns,
                              *[f"--exclude={p}" for p in SKIP_PATTERNS]], dest)
        dirs[key] = dest

    pbs_dir = cache / f"pbs-licenses-{py_version}-{pbs}"
    if not pbs_dir.is_dir():
        archive = fetch(pbs_url, cache / pbs_asset)
        extract(archive, ["--zstd", "--strip-components=2", "python/licenses"], pbs_dir)
    return dirs["qtbase"], dirs["qtwayland"], dirs["pyside"], pbs_dir


def licence_files(root: Path) -> list[tuple[str, str]]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append((str(path.relative_to(root)), read(path)))
    return files


def dist_licence_files(dist: Path, site: Path) -> list[tuple[str, str]]:
    found: dict[str, str] = {}
    licences = dist / "licenses"
    if licences.is_dir():
        for path in sorted(licences.rglob("*")):
            if path.is_file():
                found[f"{dist.name}: {path.relative_to(dist)}"] = read(path)
    for path in sorted(dist.glob("*")):
        if path.is_file() and LICENCE_FILE.match(path.name):
            found[f"{dist.name}: {path.name}"] = read(path)

    top_level = dist / "top_level.txt"
    if top_level.is_file():
        for name in read(top_level).splitlines():
            name = name.strip()
            if not name:
                continue
            candidate = site / name
            if candidate.is_dir():
                for path in sorted(candidate.rglob("*")):
                    if path.is_file() and LICENCE_FILE.match(path.name):
                        found[f"{dist.name}: {path.relative_to(site)}"] = read(path)
    return list(found.items())


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip())
    prefix, output = Path(sys.argv[1]), Path(sys.argv[2])

    repo = Path(__file__).resolve().parent.parent
    site = prefix / "lib" / "python3.13" / "site-packages"
    if not site.is_dir():
        sites = list((prefix / "lib").glob("python3.*/site-packages"))
        if len(sites) != 1:
            sys.exit(f"notices: cannot find site-packages below {prefix}")
        site = sites[0]
    py_short = site.parent.name.removeprefix("python")

    dists = {}
    for dist in sorted(site.glob("*.dist-info")):
        meta = dist_metadata(dist)
        dists[normalise(meta["name"])] = (dist, meta)

    def meta_for(name: str) -> dict:
        if normalise(name) not in dists:
            sys.exit(f"notices: no {name} distribution installed in {site}")
        return dists[normalise(name)][1]

    qt_meta, shiboken_meta = meta_for("PySide6_Essentials"), meta_for("shiboken6")
    qt_version = qt_meta["version"]
    if shiboken_meta["version"] != qt_version:
        sys.exit(
            "notices: PySide6 and shiboken6 versions differ: "
            f"{qt_version} vs {shiboken_meta['version']}"
        )

    py_bin = prefix / "bin" / f"python{py_short}"
    if not py_bin.is_file():
        sys.exit(f"notices: bundled interpreter is missing: {py_bin}")
    version_out = run([str(py_bin), "--version"], capture=True).stdout.strip()
    py_match = re.search(r"Python\s+(\d+\.\d+\.\d+)", version_out)
    if not py_match:
        sys.exit(f"notices: cannot parse CPython version from {version_out!r}")
    py_version = py_match.group(1)
    pbs = read(prefix / "BUILD").strip()
    if not re.fullmatch(r"\d{8}", pbs):
        sys.exit(f"notices: unexpected python-build-standalone release in {prefix / 'BUILD'}: {pbs!r}")

    icu_dir = site / "PySide6" / "Qt" / "lib"
    icu = icu_version(icu_dir)
    icu_parts = icu.split(".")
    out.append(PREAMBLE.format(
        qt_version=qt_version,
        qt_maj_min=".".join(qt_version.split(".")[:2]),
        icu_version=icu,
        icu_major=icu_parts[0],
        icu_minor=icu_parts[1],
        py_short=py_short,
        py_version=py_version,
        pbs_release=pbs,
    ).rstrip("\n"))
    out.append("")

    # 2. Semma itself, followed by 3. the llama.cpp licences.
    section("Semma", ("LICENSE", read(repo / "LICENSE")))
    # build/llama is scripts/build-llama.sh's output (the libraries and their licenses), which CI
    # restores from its cache without the llama.cpp source.
    llama = repo / "build" / "llama"
    llama_texts = [("llama.cpp/LICENSE", read(llama / "LICENSE.llama.cpp"))]
    for path in sorted(need_dir(llama / "licenses").iterdir()):
        if path.is_file():
            llama_texts.append((f"llama.cpp/licenses/{path.name}", read(path)))
    section("llama.cpp licences", *llama_texts)

    # 4. CPython and the python-build-standalone component licences.
    qt_base, qt_wayland, pyside, pbs_dir = source_licences(
        repo / "build" / "notices-src", qt_version, py_version, pbs
    )
    python_texts = [
        (f"python/cpython/LICENSE.txt", read(site.parent / "LICENSE.txt"))
    ]
    for path in sorted(pbs_dir.rglob("*")):
        if path.is_file():
            python_texts.append((f"python/{path.name}", read(path)))
    section("Python licences", *python_texts)

    # 5. ICU, Qt, Qt Wayland and PySide/shiboken sources at the staged version.
    icu_url = (
        f"https://raw.githubusercontent.com/unicode-org/icu/release-"
        f"{icu_parts[0]}-{icu_parts[1]}/icu4c/LICENSE"
    )
    cache = repo / "build" / "notices-src"
    icu_file = fetch(icu_url, cache / f"icu-{icu_parts[0]}.{icu_parts[1]}-LICENSE")
    section("ICU", ("icu/LICENSE", read(icu_file)))

    qt_texts: list[tuple[str, str]] = []
    for root, prefix_name in ((qt_base, "qt"), (qt_wayland, "qtwayland"), (pyside, "pyside")):
        for name, text in licence_files(root):
            qt_texts.append((f"{prefix_name}/{name}", text))
    section("Qt/PySide licence files", *qt_texts)

    # 6. Every installed Python distribution (except Semma's own licence, above).
    package_texts: list[tuple[str, str]] = []
    metadata_only = []
    for name, (dist, meta) in sorted(dists.items()):
        if name == "semma":
            continue
        label = f"{meta['name']} {meta['version']} ({meta['license'] or 'no licence metadata'})"
        if name in {"pyside6-essentials", "shiboken6"}:
            metadata_only.append(f"{label}: source {meta['source']}")
            continue
        files = dist_licence_files(dist, site)
        if not files and name in UPSTREAM_LICENCES:
            repo_url, names = UPSTREAM_LICENCES[name]
            raw = repo_url.replace("https://github.com/", "https://raw.githubusercontent.com/")
            for file in names:
                url = f"{raw}/v{meta['version']}/{file}"
                path = fetch(url, repo / "build" / "notices-src" / f"{name}-{meta['version']}-{file}")
                files.append((f"{repo_url}/blob/v{meta['version']}/{file}", read(path)))
            label = f"{meta['name']} {meta['version']} ({meta['license']}, {repo_url})"
        if not files:
            sys.exit(f"notices: {dist.name} ships no licence file; add it to UPSTREAM_LICENCES")
        package_texts.extend((f"{label}: {source.split(': ', 1)[-1]}", text)
                             for source, text in files)
    section("Python distribution licence files", *package_texts)
    if metadata_only:
        section(
            "Python distributions without bundled licence texts",
            ("Python distributions without bundled licence texts", "\n".join(metadata_only)),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(out), encoding="utf-8")
    print(f"{output}: {output.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
