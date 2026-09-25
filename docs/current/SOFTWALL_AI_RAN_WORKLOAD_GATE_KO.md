# SoftWall의 AI-RAN 작업 자격 gate — 2026-09-22

**판정:** C97–105의 Qwen prefill은 실제 GPU 수요·복구 경합을 찾는
AI-and-RAN canary다. NVIDIA의 공개 AI-RAN 사례도 LLM/VLM·video inference를
RAN과 함께 실행하는 `AI-and-RAN`을 포함하므로 Qwen이라는 작업 종류 자체를
도메인 밖이라고 보지는 않는다. 다만 현재 실험에는 운영 요청 trace·개별 SLO·
완료 가치가 없으므로 이것을 최종 workload나 공동 정책 노벨리티 근거로
승격하지 않는다. C96에서는 짧은 실제 AI 단위
일곱 개가 모두 들어가고, C98에서는 Qwen 일곱 번째 단위가 강제 실패가
없어도 끝나지 않았다. 단순히 AI 단위 길이를 바꿔 두 결과 사이를 고르는
것은 연구 기여가 아니다.

## 우선 후보와 데이터

현재 연구 질문과 직접 맞는 **1순위는 AI-and-RAN inference**다.
[NVIDIA Aerial Testbed의 공개 사례](https://docs.nvidia.com/aerial/framework/latest/text/community_showcase/allbesmart.html)는
5G L1과 같은 GPU에서 containerized video classification을 동시 실행하는
AI-and-RAN 검증을 설명한다. [SoftBank/NVIDIA 공개 field trial](https://developer.nvidia.com/blog/ai-ran-goes-live-and-unlocks-a-new-ai-opportunity-for-telcos/)도
외부 AI inference를 RAN 인프라로 dispatch하고 Llama 계열 workload를 사용한다.
따라서 Qwen prefill은 GPU 물리 canary와 축소된 LLM 대표로 사용할 수 있다.
최종 비교에서는 공개 serving trace 또는 사전 고정한 도착 과정, 요청별 SLO,
deadline 전 반환 token/value, 거부·late 요청을 포함해야 한다. 현재처럼 idle 동안
무한 backlog의 unit 수만 세는 방식은 최종 workload가 아니다.

**2순위는 AI-for-RAN dApp**이다. [Aerial Testbed dApp 구조](https://docs.nvidia.com/aerial/testbed/latest/text/product_description/index.html)는
Data Lake의 L1/L2 데이터를 같은 GPU 기반 DU의 dApp이 분석하고 RAN 제어를
보내는 구조를 명시한다. 이 경로는 AI-RAN 고유성이 더 직접적이지만, 공개된
실제 모델·요청 기한·GPU service가 확보돼야 한다. 단지 CPU xApp을 TensorRT로
옮겨 경합을 만드는 것은 허용하지 않는다.

[O-RAN SC QoE Predictor xApp](https://docs.o-ran-sc.org/projects/o-ran-sc-ric-app-qp/en/stable/overview.html)은
Traffic Steering의 요청을 받아 KPI monitor가 수집한 UE/셀 정보를 이용해
UL/DL 처리량 예측을 보낸다. 따라서 개별 UE/셀 예측을 `도착·마감·가치`가
있는 AI 작업으로 정의하기 좋은 **우선 후보**다. 단, 공식 개요가 여기서
작업별 millisecond deadline을 명시하지 않으므로 D153을 실제 xApp 기한으로
간주하지 않는다. 공개 xApp의 추론 구현이 그대로 GPU를 유의미하게 사용하는지도
별도 확인해야 한다. CPU 중심 모델을 이유 없이 GPU로 옮겨 인위적인 경합을
만들지 않는다.

[Open RAN Commercial Traffic Twinning Dataset](https://github.com/wineslab/open-ran-commercial-traffic-twinning-dataset)은
기지국·UE의 PHY/MAC KPI와 UL/DL bitrate 기록을 제공한다. 공개 설명에는
한 기지국, 8 UE, 30개 slicing/scheduling/cluster 조합과 CC-BY-SA-4.0
라이선스가 명시돼 있다. 모델·데이터 채택 전에는 실제 CSV의 표본 간격,
누락, 시간 정렬 및 예측 시점에 사용할 수 있는 feature를 확인한다.

## 이 작업을 물리 비교에 쓰기 위한 조건

1. **입력과 일반화:** 시각 `t` 이전 KPI만으로 시각 `t+Δ`의 UL/DL 처리량을
   예측한다. 동일 시계열의 인접 창을 무작위 train/test로 섞지 않고,
   experiment/시간 구간 단위로 분리한다. 실제 모델·필요한 경우 체크포인트·전처리·입력
   shape를 고정하고 held-out 오차 및 단순 지속값/이동평균 기준선을 기록한다.
2. **도착·마감·가치:** 요청 시각은 측정된 또는 명시적으로 생성한 xApp 요청
   trace에서, 마감은 traffic-steering 결정이 해당 예측을 소비할 수 있는
   시각에서 정한다. 실제 마감을 확보하지 못하면 synthetic 민감도 실험으로만
   보고한다. 늦은 예측과 거부된 요청도 offered load에 포함한다. AI 값은
   사전 정의한 적시 예측 품질/의사결정 기여이며 GPU unit 수만 세지 않는다.
3. **실제 서비스 상한:** 모델의 MPS cap, batch, 연속 burst와 PHY co-run을
   고정하고 `도착→GPU 완료 fence→결과 반환`의 mode별 경로를 새 달력에서
   자격화한다. 짧은 모델을 반복 호출해 길이만 부풀린 작업은 stress 대조로
   분리한다. 실측 수요가 모든 실패 복구와 함께 들어가면 그 작업 class에는
   공동 선택 압력이 없다고 판정한다.
4. **정책 대조:** 동일한 radio PHY trace·KPI 요청·온라인 정보와 all-fail
   안전 검사에서 low-feature gate, 최소-NRx 단계별, max-radio 단계별, 공동
   1-swap greedy 및 새 다중 사건 정책을 비교한다. 무선 비열등성과 적시
   AI 값 증가가 독립 seed·ABBA에 유지되고, PHY 가치/조건부 복구 ablation이
   이득 원인을 설명해야 성능 노벨리티를 주장한다.

Qwen을 최종 AI-and-RAN 서비스로 선택한다면 서비스 계약·실제 도착률·개별
마감이 따로 필요하다. 어느 후보든 현재 synthetic P180/D155 PHY의 통과를
실제 DU `d_MAC` 보장으로 바꾸려면 대상 DU 기한과 전송 경로를 독립 검증해야
한다. [C99 실패와 C100 새 mode](SOFTWALL_POST_C96_NOVELTY_GATE_KO.md)는
그 작업 class 변경이 bound 자격에도 영향을 준다는 근거다.
