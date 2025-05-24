# 📘 불완전한 데이터에 대한 딥러닝 모델 학습 전략 개발

> 국민대학교 소프트웨어학부 졸업 연구  
> 팀원: 임경민, 조정민, 홍선재  
> 지도교수: 김영욱 교수님  

![poster](https://github.com/user-attachments/assets/e7501489-7051-4d5e-b29f-16da6f4e7fec)

---

## 🔎 연구 배경 및 가설

최근 딥러닝 모델은 다양한 분야에서 탁월한 성능을 보이고 있으나, 대부분 **정제된 대규모 학습 데이터**에 의존합니다.  
하지만 실제 환경에서는 **노이즈가 포함된 데이터(noisy data)**가 빈번하며, 특히 **image-text retrieval**과 같은 **멀티모달 학습**에서 성능 저하의 주요 원인이 됩니다.

이에 본 연구는 noisy 환경에서도 **강건하게 학습이 가능한 전략**을 제안하며, 다음 세 가지 핵심 기법을 통해 문제를 해결하고자 합니다:

---

## 제안 기법

### 1️. Negative Learning for Noisy Labels (NLNL)
- 잘못된 대응쌍(unmatched pairs)까지 **명시적으로 학습**
- **Negative Triplet Loss**로 false correspondence와의 유사도 거리 벌리기

### 2. Hard Negative Soft-label Filter (HNSF)
- 기존 NCR의 평균 기반 soft-label 방식 개선  
- **가장 유사한 오답쌍(hard negative)**만을 기준으로 confidence 계산 → Filtering 성능 향상

### 3️. Early-Learning Regularization (ELR)
- 학습 초기의 **clean sample 기반 표현**을 유지  
- **EMA 기반 pseudo label**을 통해 noisy label에 대한 과적합 방지

---

## ⚙️ 실험 설정

- **데이터셋**: [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/)  
- **노이즈 비율**: 20% noisy correspondence 가정  
- **Baseline 모델**:
  - SCAN (ECCV 2018)
  - VSRN, IMRAM, SGR
  - NCR (NeurIPS 2021)
- **평가지표**: Recall@1, Recall@5, Recall@10 (Image→Text, Text→Image)

---

## 📊 실험 결과 요약 (Image → Text, Recall@1 기준)

| Model         | R@1 (%) |
|---------------|--------:|
| SCAN          | 59.1    |
| SGR*          | 62.8    |
| NCR           | 75.0    |
| **HNSF**      | **77.0** |
| **CAR (ELR)** | **76.3** |
| NLNL          | 20.9    |
| CAR + HNSF    | 69.7    |

> ✅ HNSF와 CAR 기법 각각은 성능 향상에 기여  
> ⚠️ CAR + HNSF 결합 시 성능 저하 → 전략적 weight 조정 필요성 제기

---

## 🧭 결론 및 향후 방향

- **HNSF**와 **CAR(ELR)**은 noisy 환경에서의 학습 안정성과 성능 향상에 기여함  
- 반면, **NLNL**은 multimodal retrieval 상황에서는 성능 저하 발생 → 추가 보완 필요  
- 향후에는 각 기법의 **조합 방식 개선** 및 **adaptive weighting** 전략 도입을 통해 더욱 robust한 학습 모델 개발 예정

---

## 📂 참고자료

- 📄 [보고서 전문 보기 (Google Drive)](https://drive.google.com/file/d/1GQwJ0iGM53gfOezzwZN2O-ZXEeBW_H20/view?usp=sharing)
- 🧠 NCR 논문: *"Learning with Noisy Correspondence for Cross-modal Matching"*  
- 🧠 NLNL 논문: *"Negative Learning for Noisy Labels" (ICCV 2019)*  
- 🧠 ELR 논문: *"Early-learning regularization prevents memorization of noisy labels" (NeurIPS 2020)*

---

