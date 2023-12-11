# Samsung AI Challenge: Camera-Invariant Domain Adaptation

## Content
1. [Augmentation](#train-data-Augmentation)
2. [DANN](#DANN)
3. [Loss](#Loss)
4. [post processing](#post-processing-mask)
5. [mask algorithm](#mask-algorithm)
6. [Result](#Result)
7. [Leaderboard](#Leaderboard)
8. [Reference](#Reference)

## train data Augmentation

![image](https://github.com/seok-AI/Domain-Adaptation/assets/85815265/d6d43aa0-fc3f-471e-80de-f9f78a751a5e)

## DANN

![image](https://github.com/seok-AI/Domain-Adaptation/assets/85815265/e65f3a97-0ea6-4053-bbbb-70ccc9290151)

## Loss

![image](https://github.com/seok-AI/Domain-Adaptation/assets/85815265/be2149e9-070b-4ad1-a387-8a0fe34a2867)


## post processing mask

![image](https://github.com/seok-AI/Domain-Adaptation/assets/85815265/001fa84b-64c9-49d1-a83c-a07357c15d20)

## mask algorithm
We devised an algorithm that reverses the mask and puts it on the image and puts the mask up to the peak of the first derivative of the intensity.
![image](https://github.com/seok-AI/Domain-Adaptation/assets/85815265/5f373553-d394-4ff8-b488-6b21cfb37ce1)

## Result
target-source-image
![image](https://github.com/seok-AI/Domain-Adaptation/assets/85815265/a7c1b4d5-3de8-496a-a25e-1c5db571b647)

## Leaderboard
### We achieved 19th place out of 682 participants!

![image](https://github.com/seok-AI/Domain-Adaptation/assets/85815265/ee34aa8a-1bd8-41df-8d53-909f7c38176b)

## Reference
https://github.com/fungtion/DANN_py3?tab=readme-ov-file
https://dacon.io/competitions/official/236132/overview/description
