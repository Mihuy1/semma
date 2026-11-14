#!/bin/sh
# On removal only, not on an upgrade (rpm passes 0 and 1, dpkg "remove" and "upgrade").
case "$1" in
  0 | remove | purge)
    if command -v systemctl > /dev/null 2>&1; then
      systemctl --global disable semma.service > /dev/null 2>&1 || true
    fi
    ;;
esac
exit 0
