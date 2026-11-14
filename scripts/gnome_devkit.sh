#!/usr/bin/env bash
# Launch a nested GNOME Shell (mutter-devkit) window inside KDE Plasma with
# Semma wired into GNOME Overview search.
#
# Why a private D-Bus session (dbus-run-session)?
#   Running GNOME Shell directly on the host Plasma session bus causes it to
#   compete with KDE for well-known names such as org.freedesktop.Notifications,
#   org.kde.StatusNotifierWatcher, and xdg-desktop-portal backends. Starting a
#   private session bus isolates the nested shell completely from Plasma, while
#   scripts/gnome_bridge.py bridges only org.gnome.Shell.SearchProvider2
#   (/io/github/Semma/SearchProvider on io.github.Semma) back to the real host session bus.
#
# Usage:
#   ./scripts/gnome_devkit.sh [extra gnome-shell args...]
#
# Prerequisites:
#   - semma daemon running on your session bus (with /io/github/Semma/SearchProvider exported)
#   - gnome-shell and mutter-devkit installed (sudo dnf install mutter-devkit)
#
# Open the Overview in the nested window and type to search with Semma.
# Close the devkit window (or press Ctrl+C) to stop both the shell and bridge.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"

if ! command -v gnome-shell >/dev/null 2>&1; then
    echo "Error: gnome-shell is not installed." >&2
    echo "Install it with: sudo dnf install gnome-shell mutter-devkit" >&2
    exit 1
fi

if ! command -v mutter-devkit >/dev/null 2>&1 && ! rpm -q mutter-devkit >/dev/null 2>&1; then
    echo "Error: mutter-devkit is not installed." >&2
    echo "Install it with: sudo dnf install mutter-devkit" >&2
    exit 1
fi

for cmd in gdbus dbus-run-session uv; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command '$cmd' not found in PATH." >&2
        exit 1
    fi
done

if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]]; then
    echo "Error: DBUS_SESSION_BUS_ADDRESS is not set." >&2
    exit 1
fi

introspect_out="$(gdbus introspect --session -d io.github.Semma -o /io/github/Semma/SearchProvider 2>&1)" || true
if [[ "$introspect_out" != *"SearchProvider2"* ]]; then
    echo "Error: io.github.Semma (/io/github/Semma/SearchProvider) does not expose SearchProvider2 on the session bus." >&2
    echo "Restart the service and try again:" >&2
    echo "  systemctl --user restart semma" >&2
    exit 1
fi

tmp_dir="$(mktemp -d)"
cleanup_tmp() {
    rm -rf -- "$tmp_dir"
}
trap cleanup_tmp EXIT INT TERM

mkdir -p \
    "$tmp_dir/share/gnome-shell/search-providers" \
    "$tmp_dir/share/applications"

cat >"$tmp_dir/share/gnome-shell/search-providers/io.github.Semma.search-provider.ini" <<'EOF'
[Shell Search Provider]
DesktopId=io.github.Semma.desktop
BusName=io.github.Semma
ObjectPath=/io/github/Semma/SearchProvider
Version=2
AutoStart=false
EOF

cat >"$tmp_dir/share/applications/io.github.Semma.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Semma
Icon=system-search
Exec=true
EOF

export SEMMA_UPSTREAM_BUS="$DBUS_SESSION_BUS_ADDRESS"

dbus-run-session -- bash -c '
set -euo pipefail
repo_dir=$1
tmp_dir=$2
shift 2

uv run --project "$repo_dir" python "$repo_dir/scripts/gnome_bridge.py" &
bridge_pid=$!

cleanup_bridge() {
    if kill -0 "$bridge_pid" 2>/dev/null; then
        kill "$bridge_pid" 2>/dev/null || true
    fi
    wait "$bridge_pid" 2>/dev/null || true
}
trap cleanup_bridge EXIT INT TERM

echo "waiting for bridge (first run may sync the uv environment)..."
ready=0
for _ in $(seq 1 300); do
    if ! kill -0 "$bridge_pid" 2>/dev/null; then
        echo "Error: gnome_bridge.py exited prematurely." >&2
        exit 1
    fi
    owner_out="$(gdbus call --session \
        -d org.freedesktop.DBus \
        -o /org/freedesktop/DBus \
        -m org.freedesktop.DBus.NameHasOwner \
        io.github.Semma 2>/dev/null || true)"
    if [[ "$owner_out" == *"(true,)"* ]]; then
        ready=1
        break
    fi
    sleep 0.1
done

if [[ "$ready" -ne 1 ]]; then
    echo "Error: timed out waiting for io.github.Semma on the private bus." >&2
    exit 1
fi

GSETTINGS_BACKEND=memory \
    XDG_DATA_DIRS="$tmp_dir/share:/usr/local/share:/usr/share" \
    XDG_CURRENT_DESKTOP=GNOME \
    gnome-shell --devkit --wayland "$@"
' _ "$repo_dir" "$tmp_dir" "$@"
