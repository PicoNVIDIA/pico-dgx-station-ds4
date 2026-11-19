#!/bin/bash
bash /raid/work/run/cells/k7_revert.sh
bash /raid/work/run/cells/k6_apply.sh || exit 1
bash /raid/work/run/cells/mkhot.sh 298
