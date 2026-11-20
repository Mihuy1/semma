#!/bin/sh
mode=
case "$1" in
  1) mode=install ;;
  configure) if [ -z "$2" ]; then mode=install; else mode=upgrade; fi ;;
  [0-9]*) mode=upgrade ;;
esac
[ -n "$mode" ] && command -v systemctl > /dev/null 2>&1 || exit 0

if [ "$mode" = install ]; then
  systemctl --global enable semma.service > /dev/null 2>&1 || true
fi

t=
command -v timeout > /dev/null 2>&1 && t="timeout 15"
as_user() { u=$1; shift; SYSTEMD_BUS_TIMEOUT=10s $t systemctl --user -M "$u@" "$@" < /dev/null 2> /dev/null; }
users() {
  [ -d /run/systemd/system ] || return 0
  $t systemctl list-units 'user@*.service' --state=active --legend=no --plain 2> /dev/null |
    sed -n 's/^user@\([0-9][0-9]*\)\.service.*/\1/p'
}
ours() {
  case $(as_user "$1" show -P FragmentPath semma.service) in
    /usr/lib/systemd/user/semma.service | /lib/systemd/user/semma.service) return 0 ;;
  esac
  return 1
}
for u in $(users); do
  as_user "$u" daemon-reload > /dev/null
  if [ "$mode" = upgrade ]; then
    ours "$u" && as_user "$u" try-restart --no-block semma.service > /dev/null
  else
    if ours "$u" && as_user "$u" is-active --quiet graphical-session.target; then
      as_user "$u" start --no-block semma.service > /dev/null
    fi
    as_user "$u" try-restart --no-block plasma-krunner.service > /dev/null
  fi
done
exit 0
