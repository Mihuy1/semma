"""D-Bus names shared by the daemon, the KRunner/GNOME plugin metadata and the install files."""

BUS_NAME = "io.github.Semma"

RUNNER_PATH = "/runner"
RUNNER_INTERFACE = "org.kde.krunner1"

CONTROL_PATH = "/control"
CONTROL_INTERFACE = "io.github.Semma.Control1"
CONFIG_ERROR = "io.github.Semma.Error.InvalidConfig"

RUNNER_ID = "semma"
SYSTEMD_UNIT = "semma.service"

GNOME_PATH = "/io/github/Semma/SearchProvider"
GNOME_INTERFACE = "org.gnome.Shell.SearchProvider2"
GNOME_DESKTOP_ID = "io.github.Semma.desktop"
