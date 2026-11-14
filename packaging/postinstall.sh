#!/bin/sh
# rpm passes 1 on a first install and 2 or more on an upgrade or reinstall; dpkg passes "configure"
# with the version it replaces, empty on a first install.
mode=
case "$1" in
  1) mode=install ;;
  configure) if [ -z "$2" ]; then mode=install; else mode=upgrade; fi ;;
  [0-9]*) mode=upgrade ;;
esac
[ -n "$mode" ] && command -v systemctl > /dev/null 2>&1 || exit 0

# On a first install only, so an upgrade does not turn the service back on where an administrator
# turned it off. Starts it with every user's graphical session from their next login (a symlink in
# /etc/systemd/user).
if [ "$mode" = install ]; then
  systemctl --global enable semma.service > /dev/null 2>&1 || true
fi

# Users logged in now, so nobody has to log in again. A stuck session must not stall the package
# manager: every call to a user's manager is bounded.
t=
command -v timeout > /dev/null 2>&1 && t="timeout 15"
as_user() { u=$1; shift; SYSTEMD_BUS_TIMEOUT=10s $t systemctl --user -M "$u@" "$@" < /dev/null 2> /dev/null; }
# The users whose systemd manager runs (logged in, or lingering), as systemd's own package scripts
# find them; none where systemd is not running (containers, image builds).
users() {
  [ -d /run/systemd/system ] || return 0
  $t systemctl list-units 'user@*.service' --state=active --legend=no --plain 2> /dev/null |
    sed -n 's/^user@\([0-9][0-9]*\)\.service.*/\1/p'
}
# Only the packaged unit: a user's own `semma install` in ~/.config takes its place and is theirs.
ours() {
  case $(as_user "$1" show -P FragmentPath semma.service) in
    /usr/lib/systemd/user/semma.service | /lib/systemd/user/semma.service) return 0 ;;
  esac
  return 1
}
for u in $(users); do
  as_user "$u" daemon-reload > /dev/null
  if [ "$mode" = upgrade ]; then
    # The new code, only where the service already runs.
    ours "$u" && as_user "$u" try-restart --no-block semma.service > /dev/null
  else
    if ours "$u" && as_user "$u" is-active --quiet graphical-session.target; then
      as_user "$u" start --no-block semma.service > /dev/null
    fi
    # KRunner reads its D-Bus plugins when it starts; D-Bus starts it again when next asked.
    as_user "$u" try-restart --no-block plasma-krunner.service > /dev/null
  fi
done
exit 0
