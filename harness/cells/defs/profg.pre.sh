#!/bin/bash
bash /raid/work/run/cells/k6_apply.sh || exit 1
bash /raid/work/run/cells/mkhot.sh 298
python3 /raid/work/run/cells/k5_apply.py || exit 1
bash -n /raid/work/gt/gracetier/tree/serve/serve.sh && echo "serve.sh syntax ok"
sudo -n rm -rf /raid/work/gt/gracetier/home/state/prof/profg
