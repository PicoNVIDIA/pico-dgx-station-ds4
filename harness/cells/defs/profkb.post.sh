#!/bin/bash
echo "K7 lines in this boot: $(docker logs ds41 2>&1 | grep -c 'K7 ')"
docker logs ds41 2>&1 | grep -E "K7 packed scales ARMED|K7 kernel variant|JIT include root" | sed -E 's/^.*(K7 packed scales ARMED|K7 kernel variant|JIT include root)/\1/' | head -n 3 | cut -c1-200
bash /raid/work/run/cells/k5_profile_post.sh profkb general 512
