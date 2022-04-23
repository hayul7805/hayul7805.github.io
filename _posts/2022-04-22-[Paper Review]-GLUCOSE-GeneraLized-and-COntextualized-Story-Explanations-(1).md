---
title: "[Paper Review] GLUCOSE-GeneraLized and COntextualized Story Explanations (1)"
excerpt: "GLUCOSE 논문에 대해 알아보자."

categories:
  - Paper review
tags:
  - \[NLP, Common sense]

permalink: /paper-review/glucose-1/
toc: true
toc\_sticky: true
date: 2022-04-22
---

![image-1](/assets/images/posts_img/paper-review-glucose/GLUCOSE-main-image.jpeg)

# 들어가며

이 논문은 Elemental Cognition이라는 AI기업 연구자들이 여럿 참여하였으며 2020 EMNLP 컨퍼런스에서 Honourable Mention Papers에 오른 논문이다. Velog에 처음 포스팅하는 논문으로 이 논문을 정한 까닭은 **상식 추론 데이터셋** 을 상당히 흥미로운 방식으로 수집한 연구이기 때문이다. 인간의 인지 심리학에 영향을 받아 사건의 인과 관계를 10차원으로 정의하고, 이에 맞춰 상식 추론 데이터셋을 만들었다니...역시 EMNLP에는 아무나 투고하는 것이 아니다.

# MCS

## Motivtion

- 사람은 무언가를 읽거나 들을 때, **암시적인 상식 추론(implicit commonsense inferences)**을 만들어 무엇이 일어났고 왜 일어났는지를 이해한다.
- 그러나 AI system은 reading comprehension이나 dialogue과 같은 태스크에 있어서 여전히 인간과 같은 상식 추론 능력을 보이지 못하고 있다.
- 그 이유는...
   - ‘상식(commonsense knowledge)’을 대규모로 휙득할 방법이 없기 때문에
   - 그러한 지식들을 최신의 AI 시스템에 통합시킬 방법이 없기 때문에

## Claims

- GlUCOSE 상식 추론 프레임워크를 통해 이러한 보틀넥을 해결할 수 있다.
- GLUCOSE 데이터셋은 사건의 인과 설명을 10가지 차원을 제공하며, 또한 구체적인 스토리와 일반화된 법칙을 제공한다.

## Significance

- 암시적인 상식(Commonsense knowledge)를 대규모로 수집할 방법을 수립했다.
- 기존의 사전학습 언어모델을 GLUCOSE에 훈련시키면 처음 보는 이야기에도 인간과 비슷한 정도의 상식 추론이 가능해진다.

# Introduction

![image-2](/assets/images/posts_img/paper-review-glucose/girl-and-car.png)

다음같은 상황을 가정해보자.

- 한 아이 앞에서 차가 방향을 틀었다.
- 그 아이가 재빨리 자전거를 돌렸다.
- 그 아이가 자전거에서 떨어졌다.
- 그 아이의 무릎이 까졌다.

이 이야기를 읽으며 사람은 문장간의 **'인과 추론'** 을 만들어낼 수 있다. 예를 들어,

- '한 아이 앞에서 차가 방향을 틀었다.'
- **그리고 그건** '그 아이가 재빨리 자전거를 돌렸다.'를 초래하고
- **그리고 그건** '그 아이가 자전거에서 떨어졌다.' 를 초래하고
- **그리고 그건** '그 아이의 무릎이 까졌다.'를 초래했다.

이러한 방식으로 말이다. **사람은 이처럼 어떻게 이야기 속 특정한 사건이 특정한 결과를 이끌었는지 묘사하는 '인과적 사슬'을 만들 수 있다**

그러나 AI system은 reading comprehension이나 dialogue과 같은 태스크에 있어서 여전히 인간과 같은 상식 추론 능력을 보이지 못하고 있다. 이는 두 가지 이유가 있는데, 첫째로 **암시적인 상식을 대규모로 획득할 길이 없으며**, 둘째로 그러한 지식을 **최신의 AI 시스템에 융합시킬 길이 없기** 때문이다.

이러한 상황을 해결하고 나온 것이 바로 GLUCOSE이다. GLUCOSE라는 이름도 바로 이 프레임워크가 AI를 위해 할 수 있는 기능을 비유적으로 보여주고 있는데, 사람의 지식 활동이 뇌 속의 글루코스 용량에 따라 좌우되는 것처럼, AI 시스템이 기본적 사고를 할 수 있도록 하는 연료가 되라는 의미에서 글루코스라고 지었다고 한다. 좋은 논문은 역시 이름도 잘 짓는다.

앞에서도 언급했지만, GLUCOSE 데이터는 아주 흥미로운 규칙으로 구축되었다. 논문의 표현을 가져오면 다음과 같다.

> S라는 짧은 이야기의 X라는 선택된 문장이 주어지면, GLUCOSE는 X와 관련된, 인간의 인지 심리학에 영향을 받은 10가지 차원의 commonsense causal explanations을 정의한다.

뿐만 아니라, GLUCOSE는 commonsense knowledge를 세상에 관한 ‘미니 이론’이라고 할 수 있는 **‘반정형 추론 법칙(semi-structured inference rules)'** 의 형태로 인코딩하고, 각각은 구체적인 이야기에 근거한다는 특징도 있다.

이게 무슨 말일까? 다음 포스팅에서 계속해보자.
