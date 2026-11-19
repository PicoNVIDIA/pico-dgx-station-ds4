#!/bin/bash
# Print what the boot that just ran says about K7.
K7=/raid/work/run/patches/K7
echo "K7 lines in the server log: $(docker logs ds41 2>&1 | grep -c 'K7 ')"
docker logs ds41 2>&1 | grep -E "K7 " | grep -vE "K7 packed scales: steps" | sed -E 's/^.*DS41_FASTPATH [a-z0-9_]+: //; s/^\(EngineCore[^]]*\] //' | head -n 8 | cut -c1-330
docker logs ds41 2>&1 | grep -E "K7 packed scales: steps" | tail -n 3 | sed -E 's/^.*DS41_FASTPATH [a-z0-9_]+: //; s/^\(EngineCore[^]]*\] //' | cut -c1-330
echo "kernel include root now:    $(docker logs ds41 2>&1 | grep -m1 'JIT include root' | sed -E 's/^.*(megamoe[^:]*: .*JIT include root.*)$/\1/' | cut -c1-260)"
echo "kernel include root before: $(cut -c1-260 "$K7/jit-root-before.txt" 2>/dev/null)"
docker logs ds41 2>&1 | grep -E "a12_memory_guard|MEMORY GUARD|memory guard" | tail -n 2 | sed -E 's/^\(EngineCore[^]]*\] //' | cut -c1-300
echo "server log error lines: $(docker logs ds41 2>&1 | grep -cE ' ERROR |Traceback')"
bash /raid/work/run/cells/k6_post.sh | tail -n 3
