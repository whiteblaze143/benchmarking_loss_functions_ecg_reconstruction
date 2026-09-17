# Paper 14: Invariant-Mechanism Discovery with repStat

## The Core Question
Models trained on one hospital (PTB-XL) almost always fail when deployed in other hospitals because they learn spurious correlations unique to that hospital's demographic. How do we find universal biological truths?

## The Math
Invariant Risk Minimization (IRM).

## The Architecture
We evaluate the classifier simultaneously across 9 totally different real-world datasets (PTB-XL, EchoNext, LUDB, RDB, ISP, Kingston-ICU, Emory-MUSE, Sunnybrook, Zhejiang). The IRM penalty forces the model to find a representation whose optimal linear classifier is mathematically identical across *all 9* datasets.

## The Mechanism (Biological Context)
This explicitly discards spurious, single-hospital correlations. If a feature works in PTB-XL but fails in the Kingston-ICU dataset, IRM deletes it. The model is forced to discover universal, biological mechanisms that hold true everywhere in the world.

## The Falsification & Control
- **Matched Control**: Empirical Risk Minimization (ERM) pooled across all domains.
- **Falsification**: We shuffle the domain assignments across the 9 datasets. This destroys the invariant structure, and the IRM advantage over ERM must disappear.
