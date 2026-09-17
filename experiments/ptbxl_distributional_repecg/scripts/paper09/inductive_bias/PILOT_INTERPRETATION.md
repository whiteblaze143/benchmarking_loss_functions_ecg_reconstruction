# Paper 09 rank-8 synthetic pilot

The first operator-dependent synthetic audit used an unrestricted 8-D latent
state while giving the model only 1--3 noiseless scalar operator responses in a
context. That cannot identify the target response for a generic new operator:
the unobserved state components remain unconstrained. Its failed recovery gate
is therefore preserved as an invalid-identifiability pilot, not evidence for or
against the architecture.

The production mechanism audit uses a fixed rank-2 latent source, which is
identifiable from the allowed context cardinality. Its recovery, mismatch, and
target-operator gates are the relevant architectural test.
