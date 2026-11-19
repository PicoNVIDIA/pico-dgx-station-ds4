#!/bin/bash
bash /raid/work/run/cells/k6_apply.sh || exit 1
bash /raid/work/run/cells/mkhot.sh 298
bash /raid/work/run/cells/k7_apply.sh || { echo "K7 NOT APPLIED: this boot proves nothing about K7"; exit 1; }
