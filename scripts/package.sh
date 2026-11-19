#!/bin/bash
# Builds dist/semma-<version>.x86_64.rpm and dist/semma_<version>_amd64.deb. Both carry everything
# Semma runs on, under /usr/lib/semma: its own Python 3.13 with the locked dependencies, and the
# llama.cpp libraries (scripts/build-llama.sh), plus the files that add it to KRunner and GNOME Shell
# for every user. The model is not inside: the service downloads it the first time it runs.
# Needs uv, curl, and podman or docker (for scripts/build-llama.sh).
#   scripts/package.sh    then: scripts/check-package.sh dist/<package>
set -euo pipefail
cd "$(dirname "$0")/.."

NFPM_VERSION=2.49.0
NFPM_SHA256=2afaa5ffb5e3493402e31e7c67a5a1fb42133e3f66fc315ca0051934125561c7
PYTHON=3.13.16
PYTHON_VERSION=${PYTHON%.*}

scripts/build-llama.sh

version=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)
stage=build/stage
prefix=$stage/usr/lib/semma
rm -rf "$stage"
mkdir -p "$prefix" "$stage/usr/bin"

# Python itself: a standalone build (it needs only glibc), without what a service never uses.
uv python install --quiet --install-dir build/python "$PYTHON"
python_src=build/python/cpython-$PYTHON-linux-x86_64-gnu
cp -a "$(readlink -f "$python_src")" "$prefix/python"
py=$prefix/python/bin/python$PYTHON_VERSION
lib=$prefix/python/lib/python$PYTHON_VERSION
# The interpreter is linked statically, so libpython is only for embedding; Tcl/Tk, IDLE, pip and the
# headers are for development.
rm -rf "$prefix"/python/{include,share} "$prefix"/python/lib/{libpython*,libtcl*,tcl*,tk*,itcl*,thread*,pkgconfig} \
  "$lib"/{idlelib,tkinter,turtledemo,ensurepip,test,pydoc_data,lib2to3,config-*,EXTERNALLY-MANAGED} \
  "$lib"/site-packages/pip* "$lib"/lib-dynload/_tkinter*

# Semma and exactly the versions in uv.lock, each checked against its hash.
rm -rf build/wheel
uv build --quiet --wheel --out-dir build/wheel
uv export --quiet --frozen --no-dev --no-emit-project --format requirements-txt -o build/requirements.txt
uv pip install --quiet --python "$py" --require-hashes -r build/requirements.txt
uv pip install --quiet --python "$py" --no-deps build/wheel/semma-"$version"-*.whl
# While every wheel is still installed (before the Qt trim), build all notices and fail if a
# licence we need is missing. Qt/PySide licences are fetched from the matching upstream sources.
"$py" -I scripts/notices.py "$prefix/python" build/THIRD_PARTY_NOTICES.txt
# The commands beside the interpreter name this build folder in their first line; /usr/bin/semma
# starts Semma instead.
find "$prefix/python/bin" -mindepth 1 ! -name "python$PYTHON_VERSION" -delete

# PySide6-Essentials carries all of Qt Quick, QML, Designer, Network, SQL, PrintSupport, translations
# and developer tools (~250 MB). Semma's window uses only Qt Widgets on Wayland and X11; keep an
# explicit allow-list (a missing path fails the build; everything else in PySide6/ and shiboken6/ is
# deleted, the dist-info and the notices built above aside). libqgtk3.so is left out so the window does not pull in GTK 3
# beside Qt; libqxdgdesktopportal.so uses the desktop portal over D-Bus instead.
qt_keep=(
  shiboken6/__init__.py
  shiboken6/_config.py
  shiboken6/_git_shiboken_module_version.py
  shiboken6/Shiboken.abi3.so
  'shiboken6/libshiboken6.abi3.so.*'
  PySide6/__init__.py
  PySide6/_config.py
  PySide6/_git_pyside_version.py
  PySide6/support
  'PySide6/libpyside6.abi3.so.*'
  PySide6/QtCore.abi3.so
  PySide6/QtGui.abi3.so
  PySide6/QtWidgets.abi3.so
  PySide6/Qt/lib/libQt6Core.so.6
  PySide6/Qt/lib/libQt6Gui.so.6
  PySide6/Qt/lib/libQt6Widgets.so.6
  PySide6/Qt/lib/libQt6WaylandClient.so.6
  PySide6/Qt/lib/libQt6XcbQpa.so.6
  PySide6/Qt/lib/libQt6DBus.so.6
  PySide6/Qt/lib/libQt6Svg.so.6
  'PySide6/Qt/lib/libicudata.so.*'
  'PySide6/Qt/lib/libicui18n.so.*'
  'PySide6/Qt/lib/libicuuc.so.*'
  PySide6/Qt/plugins/platforms/libqwayland.so
  PySide6/Qt/plugins/platforms/libqxcb.so
  PySide6/Qt/plugins/platforms/libqoffscreen.so
  PySide6/Qt/plugins/platforms/libqminimal.so
  PySide6/Qt/plugins/wayland-shell-integration/libxdg-shell.so
  PySide6/Qt/plugins/wayland-decoration-client/libadwaita.so
  PySide6/Qt/plugins/wayland-decoration-client/libbradient.so
  PySide6/Qt/plugins/platformthemes/libqxdgdesktopportal.so
  PySide6/Qt/plugins/platforminputcontexts/libcomposeplatforminputcontextplugin.so
  PySide6/Qt/plugins/platforminputcontexts/libibusplatforminputcontextplugin.so
  PySide6/Qt/plugins/iconengines/libqsvgicon.so
  PySide6/Qt/plugins/imageformats/libqsvg.so
  PySide6/Qt/plugins/imageformats/libqjpeg.so
  PySide6/Qt/plugins/imageformats/libqwebp.so
  PySide6/Qt/plugins/imageformats/libqgif.so
  PySide6/Qt/plugins/imageformats/libqico.so
)
qt_trim=$stage/qt-trim
mkdir -p "$qt_trim"
(
  cd "$lib/site-packages"
  shopt -s nullglob
  for pattern in "${qt_keep[@]}"; do
    matches=($pattern)
    if (( ${#matches[@]} == 0 )); then
      echo "missing allow-listed PySide6/shiboken6 path: $pattern" >&2
      exit 1
    fi
    cp -a --parents "${matches[@]}" "$OLDPWD/$qt_trim/"
  done
)
rm -rf "$lib/site-packages"/{PySide6,shiboken6}
cp -a "$qt_trim"/. "$lib/site-packages/"
rm -rf "$qt_trim"

# Compiled once here: /usr is read-only to the service, which would otherwise compile every module
# again each time it starts. unchecked-hash: valid whatever times the installed files get.
find "$prefix/python" -name __pycache__ -prune -exec rm -rf {} +
"$py" -m compileall -q --invalidation-mode unchecked-hash "$lib" > /dev/null

# llama.cpp, where embedder.py looks for it beside its Python.
mkdir -p "$prefix/llama"
cp -a build/llama/lib/. "$prefix/llama/"

install -m 755 packaging/semma "$stage/usr/bin/semma"
install -D -m 644 README.md "$stage/usr/share/doc/semma/README.md"
install -D -m 644 build/THIRD_PARTY_NOTICES.txt "$stage/usr/share/doc/semma/THIRD_PARTY_NOTICES.txt"
install -D -m 644 build/llama/LICENSE.llama.cpp "$stage/usr/share/doc/semma/LICENSE.llama.cpp"
install -D -m 644 "$lib/LICENSE.txt" "$stage/usr/share/doc/semma/LICENSE.python"
install -D -m 644 LICENSE "$stage/usr/share/doc/semma/LICENSE"

# The desktop files, written by the same code as `semma install` writes them for one user.
"$py" - "$stage" <<'PY'
import sys
from pathlib import Path

from semma import dbus_names as names
from semma import install

stage = Path(sys.argv[1])
python = "/usr/lib/semma/python/bin/python3.13"
files = {
    f"usr/lib/systemd/user/{names.SYSTEMD_UNIT}": install.render_systemd_unit_file(python),
    f"usr/share/dbus-1/services/{names.BUS_NAME}.service": install.render_dbus_service_file(python),
    f"usr/share/krunner/dbusplugins/{names.RUNNER_ID}.desktop": install.render_desktop_file(),
    f"usr/share/gnome-shell/search-providers/{names.BUS_NAME}.search-provider.ini":
        install.render_gnome_search_provider_ini(),
    f"usr/share/applications/{names.GNOME_DESKTOP_ID}": install.render_gnome_desktop_file(python),
    f"etc/xdg/autostart/{install.TRAY_DESKTOP_ID}": install.render_tray_autostart_file(python),
}
for name, text in files.items():
    # -I: only the package's own modules, never a file in the folder Semma is started in, the
    # user's own site-packages or PYTHONPATH.
    text = text.replace(f"{python} -m semma", f"{python} -I -m semma")
    path = stage / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
PY
find "$stage" -name __pycache__ -path '*/semma/__pycache__' -newer "$stage/usr/bin/semma" -prune -exec rm -rf {} +
"$py" -m compileall -q --invalidation-mode unchecked-hash "$lib/site-packages/semma" > /dev/null
chmod -R u=rwX,go=rX "$stage"

# The newest glibc any carried library asks for: the packages depend on at least that.
glibc=$(find "$prefix" -type f \( -name '*.so*' -o -name "python$PYTHON_VERSION" \) -exec objdump -T {} + 2>/dev/null |
  grep -o 'GLIBC_[0-9]\+\.[0-9.]\+' | sed 's/GLIBC_//' | sort -Vu | tail -1)

nfpm=build/tools/nfpm
if [[ ! -x $nfpm || "$("$nfpm" --version 2>/dev/null | head -1)" != *"$NFPM_VERSION"* ]]; then
  mkdir -p build/tools
  curl -fsSL -o build/tools/nfpm.tar.gz \
    "https://github.com/goreleaser/nfpm/releases/download/v$NFPM_VERSION/nfpm_${NFPM_VERSION}_Linux_x86_64.tar.gz"
  echo "$NFPM_SHA256  build/tools/nfpm.tar.gz" | sha256sum -c - > /dev/null
  tar -xzf build/tools/nfpm.tar.gz -C build/tools nfpm
  rm build/tools/nfpm.tar.gz
fi

mkdir -p dist
rm -f dist/semma-"$version"-*.rpm dist/semma_"$version"-*.deb
for packager in rpm deb; do
  SEMMA_VERSION=$version SEMMA_GLIBC=$glibc "$nfpm" package --config packaging/nfpm.yaml --packager "$packager" \
    --target dist/ > /dev/null
done
echo "glibc >= $glibc, $(du -sh "$stage" | cut -f1) installed"
ls -la dist/semma[-_]"$version"*
