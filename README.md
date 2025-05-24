# 📘 불완전한 데이터에 대한 딥러닝 모델 학습 전략 개발

> 국민대학교 소프트웨어학부 졸업 연구  
> 팀원 : 임경민, 조정민, 홍선재  
> 지도교수 : 김영욱 교수님  

![poster](https://github.com/user-attachments/assets/e7501489-7051-4d5e-b29f-16da6f4e7fec)

---

## 연구 배경 및 가설

최근 딥러닝 모델은 다양한 분야에서 탁월한 성능을 보이고 있으나 대부분 정제된 대규모 학습 데이터에 의존합니다.  
하지만 실제 환경에서는 노이즈가 포함된 데이터(noisy data)가 빈번하며, 특히 image-text retrieval과 같은 멀티모달 학습에서 성능 저하의 주요 원인이 됩니다.

이에 본 연구는 noisy 환경에서도 강건하게 학습이 가능한 전략을 제안하며, 다음 세 가지 핵심 기법을 통해 문제를 해결하고자 합니다:

---

## 제안 기법

### 1️. Negative Learning for Noisy Labels (NLNL)
- 잘못된 대응쌍(unmatched pairs)까지 명시적으로 학습
- Negative Triplet Loss로 false correspondence와의 유사도 거리 벌리기

### 2. Hard Negative Soft-label Filter (HNSF)
- 기존 NCR의 평균 기반 soft-label 방식 개선  
- 가장 유사한 오답쌍(hard negative)만을 기준으로 confidence 계산 → Filtering 성능 향상

### 3️. Consistency-aware Regularization (CAR)
- 학습 초기의 clean sample 기반 표현을 유지  
- EMA 기반 pseudo label을 통해 noisy label에 대한 과적합 방지

---

## 실험 설정

- **데이터셋**: [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/)  
- **노이즈 비율**: 20% noisy correspondence 가정  
- **Baseline 모델**:
  - SCAN (ECCV 2018)
  - VSRN, IMRAM, SGR
  - NCR (NeurIPS 2021)
- **평가지표**: Recall@1, Recall@5, Recall@10 (Image→Text, Text→Image)

---

## 실험 결과 요약 (HNSF가 압도적으로 Image→Text R@1에서 최고 성능(77.0)을 달성한 것이 확인)
| Methods      | I→T R@1 | R@5     | R@10    | T→I R@1 | R@5     | R@10    | Ours |
| ------------ | ------- | ------- | ------- | ------- | ------- | ------- |------|
| SCAN         | 59.1    | 83.4    | 90.4    | 36.6    | 67.0    | 77.5    |      |
| VSRN         | 58.1    | 82.6    | 89.3    | 40.7    | 67.8    | 77.0    |      |
| IMRAM        | 63.0    | 86.1    | 91.3    | 41.4    | 71.1    | 78.1    |      |
| SAF          | 51.0    | 79.3    | 87.5    | 33.1    | 64.3    | 74.6    |      |
| SGR*         | 62.8    | 86.2    | 92.2    | 44.4    | 72.3    | 78.8    |      |
| SGR-C        | 72.8    | 90.8    | 94.5    | 56.4    | 82.1    | 88.1    |      |
| NCR          | 75.0    | **93.9** | **97.5** | **58.3** | 83.0    | **89.0** |      |
| **NLNC**     | 20.9    | 49.6    | 64.8    | 36.6    | 63.1    | 73.1    | ✔️   |
| **HNSF**     | **77.0** | 93.8    | 96.9    | 57.6    | **83.2** | 88.9    | ✔️   |
| **CAR**      | 76.3    | 93.4    | 96.5    | 57.6    | 82.4    | 88.6    | ✔️   |
| **CAR+HNSF** | 69.7    | 92.3    | 95.9    | 54.3    | 81.5    | 87.9    | ✔️   |

> 제안한 **Hard Negative Soft-label Filter(HNSF)** 기법은 Image-to-Text Recall@1과 Text-to-Image Recall@5 지표에서 SOTA (State-of-the-Art) 성능을 달성하였습니다.
> **HNSF**와 **CAR** 기법 각각은 성능 향상에 기여  
> **CAR** + **HNSF** 결합 시 성능 저하 → 전략적 weight 조정 필요함

---

## 결론 및 향후 방향성

- **HNSF**와 **CAR**은 noisy 환경에서의 학습 안정성과 성능 향상에 기여함
- 반면, **NLNL**은 multimodal retrieval 상황에서는 성능 저하 발생 → 추가 보완 필요  
- 각 기법의 조합 방식 개선 및 **adaptive weighting** 전략 도입을 통해 더욱 robust한 학습 모델 개발 예정
- `추후 fine-tuning 과정을 거치면 전체적으로 성능이 소폭 향상될 것으로 예상됨`
---

## 📂 Reference

- 📄 [보고서 전문 보기 (Google Drive)](https://drive.google.com/file/d/1GQwJ0iGM53gfOezzwZN2O-ZXEeBW_H20/view?usp=sharing)
-  NCR 논문: *"Learning with Noisy Correspondence for Cross-modal Matching"*  
-  NLNL 논문: *"Negative Learning for Noisy Labels" (ICCV 2019)*  
-  ELR 논문: *"Early-learning regularization prevents memorization of noisy labels" (NeurIPS 2020)*

---

