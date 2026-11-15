"""D-Bus names shared by the daemon, the KRunner/GNOME plugin metadata and the install files."""

BUS_NAME = "io.github.Semma"

# org.kde.krunner1 implementation (KRunner and the Application Launcher search).
RUNNER_PATH = "/runner"
RUNNER_INTERFACE = "org.kde.krunner1"

# Our own control interface (status, pause/resume, rescan), used by the CLI.
CONTROL_PATH = "/control"
CONTROL_INTERFACE = "io.github.Semma.Control1"
# D-Bus error of ReloadConfig when the config file cannot be read or is not valid.
CONFIG_ERROR = "io.github.Semma.Error.InvalidConfig"

# KRunner plugin id (X-KDE-PluginInfo-Name) and systemd user unit.
RUNNER_ID = "semma"
SYSTEMD_UNIT = "semma.service"

# org.gnome.Shell.SearchProvider2 implementation (GNOME Shell overview search).
GNOME_PATH = "/io/github/Semma/SearchProvider"
GNOME_INTERFACE = "org.gnome.Shell.SearchProvider2"
# GNOME search-provider .ini and the DesktopId .desktop file it points to.
GNOME_DESKTOP_ID = "io.github.Semma.desktop"
