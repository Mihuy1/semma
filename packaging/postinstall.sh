#!/bin/sh
# On a first install only (rpm passes 1; dpkg "configure" with no earlier version), so an upgrade
# does not turn the service back on where an administrator turned it off.
# Starts the search service with every user's graphical session from their next login (a symlink in
# /etc/systemd/user).
first=
case "$1" in
  1) first=1 ;;
  configure) [ -z "$2" ] && first=1 ;;
esac
if [ -n "$first" ] && command -v systemctl > /dev/null 2>&1; then
  systemctl --global enable semma.service > /dev/null 2>&1 || true
fi
exit 0
