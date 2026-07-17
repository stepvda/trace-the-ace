#!/bin/bash
# Reassemble submission_container.zip from its committed 45MB parts.
# GitHub can't hold the 726MB zip as one file, so it's split; run this to rebuild it.
set -e
cd "$(dirname "$0")"
cat submission_container.zip.part-* > submission_container.zip
echo "Reassembled submission_container.zip ($(du -h submission_container.zip | cut -f1))"
GOT=$(shasum -a 256 submission_container.zip | cut -d' ' -f1)
WANT="43fcb69b9cf7469badaf3017314a39a10cbc3e78ecc898423fca584826e25864"
[ "$GOT" = "$WANT" ] && echo "checksum OK ($GOT)" || { echo "CHECKSUM MISMATCH! got $GOT want $WANT"; exit 1; }
