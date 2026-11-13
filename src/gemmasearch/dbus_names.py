"""D-Bus names shared by the daemon, the KRunner/GNOME plugin metadata and the install files."""

BUS_NAME = "io.github.GemmaSearch"

# org.kde.krunner1 implementation (KRunner and the Application Launcher search).
RUNNER_PATH = "/runner"
RUNNER_INTERFACE = "org.kde.krunner1"

# Our own control interface (status, pause/resume, rescan), used by the CLI.
CONTROL_PATH = "/control"
CONTROL_INTERFACE = "io.github.GemmaSearch.Control1"

# KRunner plugin id (X-KDE-PluginInfo-Name) and systemd user unit.
RUNNER_ID = "gemmasearch"
SYSTEMD_UNIT = "gemmasearch.service"

# org.gnome.Shell.SearchProvider2 implementation (GNOME Shell overview search).
GNOME_PATH = "/io/github/GemmaSearch/SearchProvider"
GNOME_INTERFACE = "org.gnome.Shell.SearchProvider2"
# GNOME search-provider .ini and the DesktopId .desktop file it points to.
GNOME_DESKTOP_ID = "io.github.GemmaSearch.desktop"
