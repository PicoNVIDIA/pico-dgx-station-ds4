#!/bin/bash
bash /raid/work/run/cells/k6_post.sh
RES=/raid/work/gt/gracetier/home/state/prof/gracetier-res; C=/raid/work/run/cells
if sudo -n cp "$RES/dump-fit302.json" "$RES/dump-fit302.npz" "$C/"; then
  sudo -n chown "$(id -u):$(id -g)" "$C/dump-fit302.json" "$C/dump-fit302.npz"
  "$HOME/.local/bin/uv" run --quiet --with numpy python "$C/fitmap.py" "$C/dump-fit302.json" "$C/dump-fit302.npz" 302 "$C/hot-302.json" "$C/hot-fit"
else
  echo "NO DUMP FILES in $RES:"; sudo -n ls -la "$RES" | tail -n 6
fi
