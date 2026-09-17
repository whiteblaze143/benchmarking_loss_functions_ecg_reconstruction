# Paper 14: Invariant-Mechanism Discovery with repStat

## The Core Question
Models trained on one hospital (like PTB-XL) almost always fail when deployed in other hospitals because they learn spurious correlations unique to that hospital's specific demographic or machines. How do we force the model to find universal biological truths?

## The Mathematical Framework: IRM
We utilize **Invariant Risk Minimization (IRM)**.

## The Architecture
We evaluate the classifier simultaneously across 9 totally different real-world datasets (PTB-XL, EchoNext, LUDB, RDB, ISP, Kingston-ICU, Emory-MUSE, Sunnybrook, Zhejiang). The IRM penalty forces the model to find a representation whose optimal linear classifier is mathematically identical across *all 9* datasets simultaneously.

## The Biological Context
This explicitly destroys spurious, single-hospital correlations. If a feature works perfectly in PTB-XL but fails in the Kingston-ICU dataset, IRM forcibly deletes it from the model's brain. The model is forced to discover universal, biological mechanisms that hold true for human hearts everywhere in the world, regardless of the recording equipment.

## Rigorous Falsification & Controls
- **Matched Control**: Empirical Risk Minimization (ERM) pooled naively across all domains.
- **Falsification (The Kill Test)**: We shuffle the domain assignments (hospital labels) across the 9 datasets during training. This destroys the invariant structure, and the IRM mathematical advantage over standard ERM must immediately disappear.
