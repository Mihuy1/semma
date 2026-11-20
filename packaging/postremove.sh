#!/bin/sh
case "$1" in
  0 | remove) ;;
  *) exit 0 ;;
esac
command -v systemctl > /dev/null 2>&1 && [ -d /run/systemd/system ] || exit 0

t=
command -v timeout > /dev/null 2>&1 && t="timeout 15"
as_user() { u=$1; shift; SYSTEMD_BUS_TIMEOUT=10s $t systemctl --user -M "$u@" "$@" < /dev/null > /dev/null 2>&1; }
for u in $($t systemctl list-units 'user@*.service' --state=active --legend=no --plain 2> /dev/null |
    sed -n 's/^user@\([0-9][0-9]*\)\.service.*/\1/p'); do
  as_user "$u" daemon-reload
  as_user "$u" try-restart --no-block plasma-krunner.service
done
exit 0
