#!/bin/sh
# On removal only, not on an upgrade (rpm passes 0 and 1, dpkg "remove" and "upgrade"; "purge"
# follows a "remove" that already did this).
case "$1" in
  0 | remove) ;;
  *) exit 0 ;;
esac
command -v systemctl > /dev/null 2>&1 && [ -d /run/systemd/system ] || exit 0

# The files are gone: the managers of the users logged in now forget the unit, and KRunner its
# plugin. Bounded, as a stuck session must not stall the package manager.
t=
command -v timeout > /dev/null 2>&1 && t="timeout 15"
as_user() { u=$1; shift; SYSTEMD_BUS_TIMEOUT=10s $t systemctl --user -M "$u@" "$@" < /dev/null > /dev/null 2>&1; }
for u in $($t systemctl list-units 'user@*.service' --state=active --legend=no --plain 2> /dev/null |
    sed -n 's/^user@\([0-9][0-9]*\)\.service.*/\1/p'); do
  as_user "$u" daemon-reload
  as_user "$u" try-restart --no-block plasma-krunner.service
done
exit 0
