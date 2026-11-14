#!/bin/sh
# Starts the search service with every user's graphical session from their next login (a symlink in
# /etc/systemd/user). Before that, the first search in KRunner or GNOME Shell starts it over D-Bus.
if command -v systemctl > /dev/null 2>&1; then
  systemctl --global enable semma.service > /dev/null 2>&1 || true
fi
exit 0
