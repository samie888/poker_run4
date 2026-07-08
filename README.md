# poker44 flow-detector miner

Behavioral flow analysis miner for Poker44 (SN126): chunk-level action-flow features feeding a compact MLP risk model.


Aggregate behavioural detector for Poker44 (Bittensor SN126).

Returns a per-chunk risk score in [0,1].
Model weights ship in `models/current.joblib`; see `model_manifest.json`.

## Data attestation
No validator-private evaluation labels are used for training. Training uses released benchmark data and local runtime features.
