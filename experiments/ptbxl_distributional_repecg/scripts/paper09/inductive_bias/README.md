# Paper 09 inductive-bias gates

These gates test the operator-conditioned set model before clinical training.
They are not evidence of diagnostic benefit.

1. **Set invariance:** permuting an identical context set must leave logits and
   counterfactual predictions unchanged to numerical tolerance.
2. **Paired recovery:** in a held-out synthetic linear operator-response world,
   a model trained on correct `(q, F(q))` pairs must beat the zero-response
   predictor on a held-out target operator.
3. **Pairing kill test:** applying the same response observations to
   different-record operators during training must fail that recovery gate.
4. **Target-operator use:** the paired model's prediction for a held-out target
   must be closer to that target than the prediction produced with a different
   target operator.
5. **No invented operator effect:** in a synthetic operator-independent world,
   changing the target operator must not create a material prediction change.

The audit is an engineering mechanism gate. Clinical claims require the later,
fold-locked matched-control evaluation.
