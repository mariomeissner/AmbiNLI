# AmbiNLI Roadmap

- [x] Hyperparameter performance comparison
  - [x] “Fast”: 128 batch, 3e-5 lr, fp16
  - [x] “Slow”: 32 batch, 1e-5 lr, fp32

- [x] AmbiNLI subsets performance comparison
  - [x] Only AmbiS
  - [x] Only AmbiM
  - [x] Only AmbiU
  - [x] Only AmbiSM
  - [x] AmbiFull
  
- [ ] Fixing uNLI
  - [ ] Check how they normalize the data in the paper
  - [ ] Find better conversion method
  - [ ] Don't include test / dev splits in AmbiNLI 
 
- [ ] Investigate about divergence calculation methods
  - [ ] Is cross-entropy really the best loss in this case?
  - [ ] What are the properties of JSD? 
  - [ ] What are the properties of KL? Should we also use it even though it is not symmetric? How does it differ from JSD?

- [ ] Test generalization performance
  - [ ] Exclude one of SNLI/MNLI completely from training, use only in test
  - [ ] ...?
  
- [ ] Study entropy averages between finetuned and un-finetuned models.
  - [ ] Do AmbiNLI-finetuned models have higher average output entropy?

- [ ] Compare performance between high and low entropy regions of ChaosNLI

- [ ] Study question stability during training
  - [ ] Amount of questions that flip answer every epoch when compared to gold-label training (like they do in Cartography)

- [ ] Can our models detect ambiguity in other datasets? Compare with Cartography?

- [ ] Evaluation set following the umbrella example.
