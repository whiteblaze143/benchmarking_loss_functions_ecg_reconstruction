# Paper 6: Conditional RepStat

## The Core Question
Does the T-wave contain *new* information, or could we have diagnosed the patient entirely just by looking at the QRS complex?

## The Math
Conditional Maximum Mean Discrepancy (Conditional MMD). Instead of measuring the raw distribution, we measure the distribution *given* prior information. 

## The Architecture
The network computes conditional KMEs. It mathematically projects the current phase state so that it is strictly orthogonal to prior states $Z_{<g}$ (or orthogonal to clinical metadata like age/sex).

## The Mechanism (Biological Context)
This is a rigorous test of conditional dependence. It prevents the model from "double-counting" information. If the QRS complex perfectly predicts the T-wave, the model is forced to recognize that the T-wave provides absolutely no *new* conditional information for the diagnosis.

## The Falsification & Control
- **Matched Control**: Marginal MMD (Paper 1).
- **Falsification**: Injecting highly correlated redundant signals. The conditional model should completely ignore the redundant signal, while the marginal model will over-index on it.
