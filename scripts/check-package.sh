#!/bin/bash
# Checks a package the way someone first meets it: installed with the distribution's own package
# manager in a bare container (no desktop, no GPU, so the model runs on the processor), the daemon
# must find every library it needs without pulling in desktop libraries, the window must find all of
# its libraries (Wayland, X11 and offscreen) once the package's recommended GUI libraries are
# installed and open cleanly with QT_QPA_PLATFORM=offscreen, and Semma must index a note and a
# picture with the real model, find both by meaning, and start its service when something asks for
# it on the session bus (as KRunner and GNOME Shell do).
#   scripts/check-package.sh dist/semma-0.6.0-1.x86_64.rpm [image]   (default: fedora:latest)
#   scripts/check-package.sh dist/semma_0.6.0-1_amd64.deb [image]    (default: ubuntu:24.04)
# The model comes from ~/.local/share/semma/models when it is there, else it is downloaded into
# build/models once. Needs podman or docker.
set -euo pipefail
cd "$(dirname "$0")/.."
package=${1:?usage: scripts/check-package.sh <package> [image]}
# install_gui: the recommends without Vulkan, ffmpeg and Tesseract (optional for the daemon, not
# libraries of the window, and OCR is off by default), so only the window's libraries come in.
case $package in
  *.rpm) image=${2:-registry.fedoraproject.org/fedora:latest}
         install="dnf install -y -q --setopt=install_weak_deps=False /pkg/$(basename "$package") dbus-daemon systemd > /dev/null"
         install_gui="dnf install -y -q --setopt=install_weak_deps=False \$(rpm -qp --recommends /pkg/$(basename "$package") | grep -vE '^(vulkan-loader|/usr/bin/ffmpeg|tesseract-libs|tesseract-langpack-eng)\$') > /dev/null"
         reinstall="rpm -U --nosignature --replacepkgs /pkg/$(basename "$package")" remove="rpm -e semma" ;;
  *.deb) image=${2:-docker.io/library/ubuntu:24.04}
         install="apt-get update -q > /dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
           --no-install-recommends /pkg/$(basename "$package") dbus-daemon systemd > /dev/null"
         install_gui="DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends \$(dpkg-deb -f /pkg/$(basename "$package") Recommends | tr ',' '\n' | grep -vE '(libvulkan1|ffmpeg|tesseract)') > /dev/null"
         reinstall="dpkg -i /pkg/$(basename "$package")" remove="dpkg --purge semma" ;;
  *) echo "not an .rpm or a .deb: $package" >&2; exit 1 ;;
esac
engine=$(command -v podman || command -v docker) || { echo "needs podman or docker" >&2; exit 1; }

models=${XDG_DATA_HOME:-$HOME/.local/share}/semma/models
mode=ro
if ! ls "$models"/*.gguf > /dev/null 2>&1; then
  models=$PWD/build/models
  mode=rw
  mkdir -p "$models"
fi

echo "checking $package on $image"
"$engine" run --rm --security-opt label=disable -v "$(realpath "$(dirname "$package")"):/pkg:ro" \
  -v "$models:/models:$mode" \
  -e SEMMA_MODEL_DIR=/models -e install_gui="$install_gui" -e reinstall="$reinstall" -e remove="$remove" "$image" bash -c "
set -euo pipefail
$install"'
# The licenses of everything the package carries (scripts/notices.py). A .deb also has them where
# Debian looks for them, in copyright: the only file under /usr/share/doc that minimized systems
# (such as the Ubuntu container here) install, so on Debian/Ubuntu that is what is checked.
doc=/usr/share/doc/semma
if command -v dpkg > /dev/null; then
  notices=$doc/copyright
else
  notices=$doc/THIRD_PARTY_NOTICES.txt
  for file in $doc/{LICENSE,LICENSE.python,LICENSE.llama.cpp}; do
    test -s "$file" || { echo "missing license notice: $file"; exit 1; }
  done
fi
test -s "$notices" || { echo "missing license notices: $notices"; exit 1; }
for text in "GNU LESSER GENERAL PUBLIC LICENSE" "Version 3, 29 June 2007" "PYTHON SOFTWARE FOUNDATION LICENSE" \
    "Unicode" "ggml authors" "sqlite-vec"; do
  grep -qF "$text" "$notices" || { echo "notices lack \"$text\""; exit 1; }
done
for dist in pyside6_essentials shiboken6; do
  test -s /usr/lib/semma/python/lib/python3.13/site-packages/"$dist"-*.dist-info/METADATA ||
    { echo "missing $dist metadata"; exit 1; }
done
# Without recommends (a minimal or headless install), everything the daemon and CLI use must
# already find its libraries; only PySide6 waits for the recommended desktop libraries.
missing=$(find /usr/lib/semma \( -path "*/site-packages/PySide6" -o -path "*/site-packages/shiboken6" \) -prune \
  -o -type f \( -name "*.so*" -o -name python3.13 \) -exec ldd {} + 2>&1 |
  grep "not found" | grep -v libvulkan.so.1 | sort -u || true)   # Vulkan is optional
if [[ -n $missing ]]; then echo "missing daemon libraries:"; echo "$missing"; exit 1; fi
test -L /etc/systemd/user/graphical-session.target.wants/semma.service || { echo "the service is not enabled for users"; exit 1; }
# An upgrade (here the same version again) leaves the service off where an administrator turned it off.
systemctl --global disable semma.service 2> /dev/null
$reinstall > /dev/null
test ! -e /etc/systemd/user/graphical-session.target.wants/semma.service || { echo "an upgrade enabled the service again"; exit 1; }
systemctl --global enable semma.service 2> /dev/null
for file in /usr/share/dbus-1/services/io.github.Semma.service /usr/lib/systemd/user/semma.service \
    /usr/share/krunner/dbusplugins/semma.desktop /usr/share/applications/io.github.Semma.desktop \
    /usr/share/gnome-shell/search-providers/io.github.Semma.search-provider.ini \
    /etc/xdg/autostart/io.github.Semma.Tray.desktop; do
  test -f "$file" || { echo "missing $file"; exit 1; }
done
grep -qx "Exec=/usr/lib/semma/python/bin/python3.13 -I -m semma gui" /usr/share/applications/io.github.Semma.desktop ||
  { echo "desktop entry does not start semma gui"; exit 1; }
grep -qx "Exec=/usr/lib/semma/python/bin/python3.13 -I -m semma tray" /etc/xdg/autostart/io.github.Semma.Tray.desktop ||
  { echo "tray autostart entry does not start semma tray"; exit 1; }

# With the recommended GUI libraries installed, every carried library and plugin (including the
# X11 and Wayland platform plugins) must resolve, and `semma gui` must open and render cleanly.
eval "$install_gui"
missing=$(find /usr/lib/semma -type f \( -name "*.so*" -o -name python3.13 \) -exec ldd {} + 2>&1 |
  grep "not found" | grep -v libvulkan.so.1 | sort -u || true)
if [[ -n $missing ]]; then echo "missing libraries:"; echo "$missing"; exit 1; fi
for plugin in /usr/lib/semma/python/lib/python3.13/site-packages/PySide6/Qt/plugins/platforms/libq{xcb,wayland,offscreen}.so; do
  test -f "$plugin" || { echo "missing $plugin"; exit 1; }
  if ldd "$plugin" 2>&1 | grep -q "not found"; then echo "unmet dependencies in $plugin"; ldd "$plugin"; exit 1; fi
done
semma gui --help > /dev/null
QT_QPA_PLATFORM=offscreen /usr/lib/semma/python/bin/python3.13 -I -c "
import os, sys, traceback
import PySide6.QtCore, PySide6.QtGui, PySide6.QtWidgets
from PySide6.QtCore import QLibraryInfo, QTimer
from semma.gui import app as gui_app, main as gui_main
from semma.gui.onboarding import OnboardingWindow
assert QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath).startswith(\"/usr/lib/semma/\")
# Qt only prints an exception raised in a slot and keeps running, so a failed check must end the
# process itself, and a window that never settles must not hang the job.
def fail(*exc):
    traceback.print_exception(*exc)
    sys.stderr.flush()
    os._exit(1)
sys.excepthook = fail
def timed_out():
    print(\"window: timed out\", file=sys.stderr, flush=True)
    os._exit(1)
app = gui_app.make_app()
QTimer.singleShot(120_000, timed_out)
def visible(kind):
    return [w for w in app.topLevelWidgets() if isinstance(w, kind) and w.isVisible()]
def grab(win):
    app.processEvents()
    pix = win.grab()
    assert not pix.isNull() and pix.width() > 0 and pix.height() > 0
    return f\"{pix.width()}x{pix.height()}\"
def check_setup():
    # A fresh install has not been set up, so the setup window opens first; finishing it opens the
    # main window in its place.
    wizards = visible(OnboardingWindow)
    assert len(wizards) == 1 and not visible(gui_app.MainWindow), app.topLevelWidgets()
    print(f\"setup window: ok ({grab(wizards[0])})\", flush=True)
    wizards[0]._complete()  # what a finished setup does once its settings are saved
    QTimer.singleShot(50, check_main)
def check_main():
    wins = visible(gui_app.MainWindow)
    assert len(wins) == 1 and not visible(OnboardingWindow), app.topLevelWidgets()
    win = wins[0]
    if win.status_page.running is None:
        QTimer.singleShot(50, check_main)
        return
    assert win.status_page.running is False, win.status_page.running
    win.tabs.setCurrentIndex(1)
    print(f\"window: ok ({grab(win)}, {len(win.settings_page.sections)} settings sections)\", flush=True)
    app.quit()
QTimer.singleShot(100, check_setup)
raise SystemExit(gui_main())"

mkdir -p ~/Documents
cat > ~/Documents/recipe.txt <<TEXT
Lemon cake

Cream the butter and sugar, beat in three eggs, then fold in the flour, the zest of two lemons and a splash of milk.
Bake for forty minutes at 180 degrees and pour the lemon syrup over it while it is still warm.
TEXT
cat > ~/Documents/meeting.txt <<TEXT
Quarterly budget review

The marketing team overspent on conference travel, so the remaining funds move to the hardware refresh in spring.
TEXT
/usr/lib/semma/python/bin/python3.13 -c "
from PIL import Image, ImageDraw
im = Image.new(\"RGB\", (640, 480), (40, 90, 200)); d = ImageDraw.Draw(im)
d.ellipse((220, 140, 420, 340), fill=(250, 220, 30))
im.save(\"/root/Documents/sun.png\")"

semma scan > /dev/null
semma index > /dev/null
semma status
[[ $(semma search "how long to bake the cake" -n 1 --json) == *recipe.txt* ]] ||
  { echo "search did not find recipe.txt"; exit 1; }
# Meaning, not words (three files are too few for the search to rank by meaning alone): each
# question is closest to its own file, and the picture to its own description.
/usr/lib/semma/python/bin/python3.13 -c "
from pathlib import Path
from PIL import Image
from semma.config import load_config
from semma.embedder import Embedder
docs = Path.home() / \"Documents\"
e = Embedder(load_config())
names = [\"recipe.txt\", \"meeting.txt\"]
texts = e.embed_batch(\"text\", [(docs / n).read_text() for n in names], names)
image = e.embed_batch(\"image\", [Image.open(docs / \"sun.png\").convert(\"RGB\")], [\"sun.png\"])[0]
cake = e.embed_query(\"dessert baking instructions\")
money = e.embed_query(\"who spent too much money\")
circle = e.embed_query(\"a yellow circle on a blue background\")
assert (texts @ cake).argmax() == 0 and (texts @ money).argmax() == 1, (texts @ cake, texts @ money)
assert image @ circle > max(image @ cake, image @ money), (image @ circle, image @ cake, image @ money)
print(\"meaning: ok on\", e.device)
e.unload()"

# The session bus starts the service from its .service file, as it does for KRunner and GNOME Shell.
dbus-run-session -- /usr/lib/semma/python/bin/python3.13 -c "
from semma.daemon import daemon_call
status = daemon_call(\"Status\", auto_start=True, timeout=60)
print(\"service: started over D-Bus,\", status.get(\"state\", status))"
$remove > /dev/null
test ! -e /etc/systemd/user/graphical-session.target.wants/semma.service || { echo "removing left the service enabled"; exit 1; }
echo ok'
