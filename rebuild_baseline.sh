#!/bin/bash
# Auto-rebuild baseline after package updates

BASELINE="/opt/trustgate/known_good_hashes_ima.txt"
LOG="/opt/trustgate/baseline_changes.log"

echo "[$(date)] AUTO-REBUILD triggered by package update" >> "$LOG"

# Backup old baseline
cp "$BASELINE" "${BASELINE}.backup.$(date +%Y%m%d_%H%M%S)"

# Rebuild from current system
find /bin /sbin /usr/bin /usr/sbin /lib /usr/lib \
    -type f \
    -exec sha256sum {} \; 2>/dev/null > "${BASELINE}.new"

# Replace
mv "${BASELINE}.new" "$BASELINE"

# Clear pending approvals
> /opt/trustgate/pending_approval.txt

echo "[$(date)] AUTO-REBUILD complete - $(wc -l < "$BASELINE") entries" >> "$LOG"
