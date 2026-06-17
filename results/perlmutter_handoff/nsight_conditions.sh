# Shared no-MIG AI condition set for all Nsight runs (default/MPS x nsys/ncu).
# Matches the AI-type dimension of the CloudLab S2-S26 NCU/nsys campaign
# (partition dimension collapses on full-GPU no-MIG).
# The sourcing script must define: ai_bg_venv, ai_bg_base, kill_all_ai, log.

CONDS_FULL="alone qwen xapp chanpred forecaster neuralrx resnet sat_compute sat_hbm resnet_fore 2AI 2sat 3AI 3sat"

# Launch the background AI workload(s) for a condition (nothing for 'alone').
launch_condition() {
  case "$1" in
    alone)        ;;
    qwen)         ai_bg_venv qwen run_qwen_small_stress.py ;;
    xapp)         ai_bg_venv xapp run_xapp_anomaly.py ;;
    chanpred)     ai_bg_venv chanpred run_channel_prediction.py ;;
    forecaster)   ai_bg_venv forecaster run_traffic_forecaster.py 64 384 ;;
    neuralrx)     ai_bg_base neuralrx run_neural_rx_stress.py ;;
    resnet)       ai_bg_venv resnet run_resnet_stress.py 64 fp16 ;;
    sat_compute)  ai_bg_venv satc run_realistic_ai_stress.py matmul 0.8 ;;
    sat_hbm)      ai_bg_venv sath run_hbm_stress.py 16 ;;
    resnet_fore)  ai_bg_venv rf_r run_resnet_stress.py 64 fp16; ai_bg_venv rf_f run_traffic_forecaster.py 64 384 ;;
    2AI)          ai_bg_venv a1 run_channel_prediction.py; ai_bg_venv a2 run_resnet_stress.py 64 fp16 ;;
    2sat)         ai_bg_venv s1 run_hbm_stress.py 16; ai_bg_venv s2 run_realistic_ai_stress.py matmul 0.8 ;;
    3AI)          ai_bg_venv a1 run_channel_prediction.py; ai_bg_venv a2 run_resnet_stress.py 64 fp16; ai_bg_venv a3 run_qwen_small_stress.py ;;
    3sat)         ai_bg_venv s1 run_hbm_stress.py 16; ai_bg_venv s2 run_realistic_ai_stress.py matmul 0.8; ai_bg_venv s3 run_realistic_ai_stress.py video 0.8 ;;
    *)            log "  !! unknown condition $1"; return 1 ;;
  esac
}

# Settle time (s) before measuring: NeuralRx needs TRT build; stacks need more.
settle_for() {
  case "$1" in
    alone)            echo 0 ;;
    neuralrx)         echo "${NEURALRX_WAIT:-75}" ;;
    2AI|2sat|3AI|3sat|resnet_fore) echo 14 ;;
    qwen)             echo 15 ;;
    *)                echo 10 ;;
  esac
}
