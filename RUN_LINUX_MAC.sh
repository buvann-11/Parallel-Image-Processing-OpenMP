#!/bin/sh
# Double-click (or run ./RUN_LINUX_MAC.sh) to open the project menu.
# Pass a command to skip the menu:  ./RUN_LINUX_MAC.sh all
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
    exec python3 run_project.py "$@"
fi
echo "Python 3 is required -- see REQUIREMENTS.md"
exit 1
