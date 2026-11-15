#!/bin/bash
# Checks a package the way someone first meets it: installed with the distribution's own package
# manager in a bare container (no desktop, no GPU, so the model runs on the processor), the daemon
# must find every library it needs without pulling in desktop libraries, the window must find all of
# its libraries (Wayland, X11 and offscreen) once the package's recommended GUI libraries are
# installed and open cleanly with QT_QPA_PLATFORM=offscreen, and Semma must index a note and a
# picture with the real model, find both by meaning, and start its service when something asks for
# it on the session bus (as KRunner and GNOME Shell do).
#   scripts/check-package.sh dist/semma-0.2.0-1.x86_64.rpm [image]   (default: fedora:latest)
#   scripts/check-package.sh dist/semma_0.2.0-1_amd64.deb [image]    (default: ubuntu:24.04)
# The model comes from ~/.local/share/semma/models when it is there, else it is downloaded into
# build/models once. Needs podman or docker.
set -euo pipefail
cd "$(dirname "$0")/.."
package=${1:?usage: scripts/check-package.sh <package> [image]}
case $package in
  *.rpm) image=${2:-registry.fedoraproject.org/fedora:latest}
         install="dnf install -y -q --setopt=install_weak_deps=False /pkg/$(basename "$package") dbus-daemon systemd > /dev/null"
         install_gui="dnf install -y -q --setopt=install_weak_deps=False \$(rpm -qp --recommends /pkg/$(basename "$package") | grep -vE '^(vulkan-loader|/usr/bin/ffmpeg)\$') > /dev/null"
         reinstall="rpm -U --nosignature --replacepkgs /pkg/$(basename "$package")" remove="rpm -e semma" ;;
  *.deb) image=${2:-docker.io/library/ubuntu:24.04}
         install="apt-get update -q > /dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
           --no-install-recommends /pkg/$(basename "$package") dbus-daemon systemd > /dev/null"
         install_gui="DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends \$(dpkg-deb -f /pkg/$(basename "$package") Recommends | tr ',' '\n' | grep -vE '(libvulkan1|ffmpeg)') > /dev/null"
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
    /usr/share/gnome-shell/search-providers/io.github.Semma.search-provider.ini; do
  test -f "$file" || { echo "missing $file"; exit 1; }
done
grep -qx "Exec=/usr/lib/semma/python/bin/python3.13 -I -m semma gui" /usr/share/applications/io.github.Semma.desktop ||
  { echo "desktop entry does not start semma gui"; exit 1; }

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
import PySide6.QtCore, PySide6.QtGui, PySide6.QtWidgets
from PySide6.QtCore import QLibraryInfo, QTimer
from semma.gui import app as gui_app, main as gui_main
assert QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath).startswith(\"/usr/lib/semma/\")
app = gui_app.make_app()
def check():
    wins = [w for w in app.topLevelWidgets() if isinstance(w, gui_app.MainWindow) and w.isVisible()]
    assert len(wins) == 1, wins
    win = wins[0]
    if win.status_page.running is None:
        QTimer.singleShot(50, check)
        return
    assert win.status_page.running is False, win.status_page.running
    win.tabs.setCurrentIndex(1)
    app.processEvents()
    pix = win.grab()
    assert not pix.isNull() and pix.width() > 0 and pix.height() > 0
    print(f\"window: ok ({pix.width()}x{pix.height()}, {len(win.settings_page.sections)} settings sections)\")
    app.quit()
QTimer.singleShot(100, check)
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
