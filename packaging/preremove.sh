#!/bin/sh
# On removal only, not on an upgrade (rpm passes 0 and 1, dpkg "remove" and "upgrade").
case "$1" in
  0 | remove | purge) ;;
  *) exit 0 ;;
esac
command -v systemctl > /dev/null 2>&1 || exit 0
systemctl --global disable semma.service > /dev/null 2>&1 || true

# Stops the service of every user logged in now, while its files are still there; postremove.sh
# reloads their managers once the files are gone. Bounded, as a stuck session must not stall the
# package manager.
t=
command -v timeout > /dev/null 2>&1 && t="timeout 15"
as_user() { u=$1; shift; SYSTEMD_BUS_TIMEOUT=10s $t systemctl --user -M "$u@" "$@" < /dev/null 2> /dev/null; }
# The users whose systemd manager runs (logged in, or lingering), as systemd's own package scripts
# find them; none where systemd is not running (containers, image builds).
[ -d /run/systemd/system ] || exit 0
for u in $($t systemctl list-units 'user@*.service' --state=active --legend=no --plain 2> /dev/null |
    sed -n 's/^user@\([0-9][0-9]*\)\.service.*/\1/p'); do
  # Only the packaged unit: a user's own `semma install` in ~/.config takes its place and is theirs.
  case $(as_user "$u" show -P FragmentPath semma.service) in
    /usr/lib/systemd/user/semma.service | /lib/systemd/user/semma.service)
      as_user "$u" stop semma.service > /dev/null ;;
  esac
done
exit 0
