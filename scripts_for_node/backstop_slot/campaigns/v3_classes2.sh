#!/usr/bin/env bash
# Step 5 rerun after fixing the admission estimate of ai_worker3 (token-based).  All
# policies are rerun so they use the same worker code (tags w?... -> z?...).
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab
sed 's/go w\${seed}/go z${seed}/g' run_state/backstop_slot/v3_classes.sh > run_state/backstop_slot/v3_classes_z.sh
bash run_state/backstop_slot/v3_classes_z.sh
sed 's/go x\${seed}/go v${seed}/g' run_state/backstop_slot/v3_contexts.sh > run_state/backstop_slot/v3_contexts_v.sh
bash run_state/backstop_slot/v3_contexts_v.sh 24 6.5 2.8 "128:3.5,512:3.9,1024:4.1" "conv=128,512,1024" 16 32
echo CLASSES2_DONE
