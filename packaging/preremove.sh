#!/bin/sh
case "$1" in
  0 | remove | purge) ;;
  *) exit 0 ;;
esac
command -v systemctl > /dev/null 2>&1 || exit 0
systemctl --global disable semma.service > /dev/null 2>&1 || true

t=
command -v timeout > /dev/null 2>&1 && t="timeout 15"
as_user() { u=$1; shift; SYSTEMD_BUS_TIMEOUT=10s $t systemctl --user -M "$u@" "$@" < /dev/null 2> /dev/null; }
[ -d /run/systemd/system ] || exit 0
for u in $($t systemctl list-units 'user@*.service' --state=active --legend=no --plain 2> /dev/null |
    sed -n 's/^user@\([0-9][0-9]*\)\.service.*/\1/p'); do
  case $(as_user "$u" show -P FragmentPath semma.service) in
    /usr/lib/systemd/user/semma.service | /lib/systemd/user/semma.service)
      as_user "$u" stop semma.service > /dev/null ;;
  esac
done
exit 0
