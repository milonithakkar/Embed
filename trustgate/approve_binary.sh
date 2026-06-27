#!/bin/bash
INPUT="$1"
BASELINE="/opt/trustgate/known_good_hashes.txt"
CHANGELOG="/opt/trustgate/baseline_changes.log"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <filepath_or_package_name>"
    exit 1
fi

if [[ -f "$INPUT" ]]; then
    HASH=$(sha256sum "$INPUT" | awk '{print $1}')
    echo "$HASH  $INPUT" >> "$BASELINE"
    echo "[$(date)] APPROVED: $INPUT" >> "$CHANGELOG"
    echo "✅ Added to baseline"
    exit 0   # <-- this exit was likely missing, causing fallthrough


dpkg -L "$INPUT" | while read filepath; do
    if [[ -f "$filepath" ]]; then
        HASH=$(sha256sum "$filepath" 2>/dev/null | awk '{print $1}')
        if [[ -n "$HASH" ]]; then
            # Only add if not already in baseline
            if ! grep -qF "$filepath" "$BASELINE"; then
                echo "$HASH  $filepath" >> "$BASELINE"
                echo "  ✅ $filepath"
            else
                echo "  ⏭️  $filepath (already in baseline)"
            fi
        fi
    fi
done

elif dpkg -l "$INPUT" 2>/dev/null | grep -q '^ii'; then
    echo "Approving package: $INPUT"
    dpkg -L "$INPUT" | while read filepath; do
        if [[ -f "$filepath" ]]; then
            HASH=$(sha256sum "$filepath" 2>/dev/null | awk '{print $1}')
            if [[ -n "$HASH" ]]; then
                echo "$HASH  $filepath" >> "$BASELINE"
                echo "  OK: $filepath"
            fi
        fi
    done
    echo "[$(date)] APPROVED PACKAGE: $INPUT" >> "$CHANGELOG"

else
    echo "ERROR: '$INPUT' not found"
    exit 1
fi

echo "Baseline now has $(wc -l < $BASELINE) entries"
