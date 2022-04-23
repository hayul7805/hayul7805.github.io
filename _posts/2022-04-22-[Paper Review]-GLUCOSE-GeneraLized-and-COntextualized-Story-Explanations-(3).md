---
title: "Paper Review: GLUCOSE-GeneraLized and COntextualized Story Explanations (3)"
excerpt: "GLUCOSE 논문에 대해 알아보자."

categories:
  - Machine Learning
tags:
  - \[NLP, Common sense]

permalink: /machine-learning/glucose-3/
toc: true
toc\_sticky: true
date: 2022-04-23
---
# Emperical evaluation task

> **Q2. How to incorporate commonsense knowledge into the state-of-the-art AI systems?**
>   지난 포스팅에서는 데이터 수집 방법론을 다뤘다. 이번에는 그렇게 해서 구축한 데이터셋을 가지고 어떤 일을 할 수 있을지 다뤄보려고 한다.

본 연구에서는 GLUCOSE 데이터를 가지고 모델을 평가할 수 있는 `evaluation task` 를 만들었다. 구체적으로 평가는 다음과 같다.

> Task: S라는 짧은 이야기의 X라는 선택된 문장이 주어지고, 차원 d가 주어지면, specific and general forms의 explanation을 생성하기

실험에 쓰이는 데이터셋은 앞서 다룬 파이프라인에서 성적이 높은 Best worker들이 동의하면서, 이전에 모델에게 보여주지 않은(unseen) 이야기들로 선별되었다. 이 과정으로 **총 500개의 이야기-문장(X) 쌍**을 만들었고, 이들은 1에서 5까지의 차원으로 설명되어 있다(6-10차원을 포함시키지 않은 까닭은 1-5차원과 6-10차원이 단지 문장X의 *앞* 과 *뒤* 라는 차이밖에 없기 때문으로 보인다).

Evaluation은 `human evaluation` 과 `automatic evaluation` 을 모두 사용했다. 먼저 `human evaluation` 은 MTurk에서 진행되며, 3명의 Best worker들이 참여한다. 이들은 강조된 문장X을 먼저 읽고, X에 관한 GLUCOSE dimension에 대응하는 질문들을 읽은 후, 다른 시스템들에 의해 생성된 list of candidate answers을 보고, 각 candidate answer에 대해 `four-point Likert scale` 로 채점하게 된다.

다음으론 `automatic evaluation` 이다. Metric으로는 `BLUE score` 를 사용하였는데, 사실 BLUE score는 손쉬운 재현성으로 유명한 메트릭이 되었지만, <mark>인간의 결과와의 상관관계에 있어서 여러 태스크들에서 약한 모습을 보인다</mark>는 약점이 있다. 이에 따라 먼저 human evaluation과의 상관관계를 살펴보았다. 본 연구에서 사용한 `pairwise correlation analysis` 의 결과를 보자면, **사람의 결과와 ScareBLEU 스코어 사이에 강한 correlation이 있음을 발견했다**(Spearman = 0.891, Pearson = 0.855, and Kendall’s τ = 0.705, all with p-value < 0.001). 이에 저자들은 BLUE score로 test set의 결과를 report하는 데 쓰일 수 있다고 판단하였고, `SacreBLUE score` 를 최종 metric으로 선정했다. BLUE score에 대한 자세한 설명은 [BLUE score](/machine-learning/blue-score/)게시글을 참조해보자.

