# Confusion Matrix Analysis: Canonical Protocol12 v1-v3

Date: 2026-07-06

Scope:
- PAMAP2 protocol-only / protocol12
- Canonical result sets:
  - `canonical_protocol_only`
  - `canonical_protocol_only_v2`
  - `canonical_protocol_only_v3`
- Main focus: LOSO confusion matrices from best deep models.

## Current Best LOSO Models

Best current overall model remains:

```text
canonical_protocol_only / acc16_gyro_hr / improved_gnn_lstm_attn_adj
macro_f1 = 0.885490
accuracy = 0.889169
balanced_accuracy = 0.884160
```

Best v3 result:

```text
canonical_protocol_only_v3 / acc16_gyro / improved_gnn_lstm_attn_adj_resbn
macro_f1 = 0.880979
accuracy = 0.877914
balanced_accuracy = 0.876566
```

Conclusion: v3 helps `acc16_gyro`, but it does not beat the v1 `acc16_gyro_hr` attention-adjacency model.

## Recurrent Confusion Patterns

### 1. Static Posture Confusion

Best v1 `acc16_gyro_hr + improved_gnn_lstm_attn_adj`:

```text
sitting -> standing: 542 / 2882 = 0.188
standing -> sitting: 367 / 2956 = 0.124
```

This is the largest remaining error family. It is likely driven by subject-specific sensor orientation and posture similarity.

### 2. Stair Direction / Locomotion Confusion

Best v1:

```text
ascending_stairs -> descending_stairs: 102 / 1808 = 0.056
nordic_walking -> walking: 165 / 2928 = 0.056
```

v3 residual `acc16_gyro_hr`:

```text
walking -> ascending_stairs: 514 / 3717 = 0.138
ascending_stairs -> descending_stairs: 144 / 1808 = 0.080
descending_stairs -> ascending_stairs: 139 / 1614 = 0.086
```

This suggests the model needs better temporal shape and frequency/direction cues, not just a larger graph.

### 3. Household/Repetitive Motion Confusion

Best v1:

```text
vacuum_cleaning -> ironing: 85 / 2728 = 0.031
ironing -> vacuum_cleaning: 84 / 3718 = 0.023
ironing -> cycling: 116 / 3718 = 0.031
```

These are repetitive-arm-motion classes. A pure statistical node summary loses some waveform and rhythm information.

### 4. Running / Rope-Jumping Confusion

v3 attention-adjacency with HR:

```text
running -> rope_jumping: 238 / 1526 = 0.156
```

This is a vertical-impact / high-energy rhythm confusion.

## Subject 103 and 109 Caveat

Subject 103 has only 8 of 12 protocol classes:

```text
windows = 2708
classes = 8
```

Subject 109 has only one class:

```text
windows = 98
class = rope_jumping only
sequence predictions = 89 with seq_len=10
```

For subject 109, perfect accuracy can still produce macro-F1 around `1 / 12 = 0.0833` because macro-F1 is computed over all protocol classes. Treat subject 109 macro-F1 as a label-coverage artifact, not a normal failure fold.

## Why Larger v3 Is Not Sufficient

The current PAMAP2 graph path converts each raw window to node summaries:

```text
mean, std, min, max, rms, iqr + categorical node context
```

That is useful, but it discards raw temporal waveform shape inside each 128-sample window. The remaining confusion pairs are exactly the kinds of mistakes that need waveform rhythm, frequency, slope, and phase information:

- sitting vs standing: posture/orientation differences
- walking vs stairs: gait shape and vertical acceleration
- running vs rope jumping: periodic impact structure
- ironing/vacuum/cycling: repetitive motion rhythm

Adding more residual graph layers increases capacity, but it does not restore lost temporal information.

## Proposed v4 Direction

Do not keep making the GNN wider as the main fix. Build a v4 model with:

1. Raw temporal node encoder
   - For each graph node/window, encode raw 128-sample signal using a small TCN/CNN branch before graph message passing.
   - Keep statistical features as auxiliary inputs, but do not make them the only node representation.

2. Hybrid node fusion
   - Concatenate or gate:
     - raw temporal embedding
     - statistical node features
     - node context embedding

3. Graph message passing after temporal encoding
   - Apply attention-adjacency graph layers to temporally encoded node embeddings.

4. Sequence model across windows
   - Keep the 10-window sequence LSTM/attention path.
   - Consider a light temporal transformer/TCN only after the node-level temporal encoder is added.

5. Hierarchical / confusion-aware heads
   - Add a coarse activity-group head:
     - posture: lying, sitting, standing
     - locomotion: walking, running, cycling, nordic_walking
     - stairs: ascending_stairs, descending_stairs
     - household: vacuum_cleaning, ironing
     - jump: rope_jumping
   - Keep the fine 12-class head.
   - Train with combined loss:
     - fine cross-entropy
     - coarse cross-entropy
     - optional confusion-pair margin/contrastive loss for known hard pairs.

6. Metric/reporting fix
   - Add present-class macro-F1 to fold summaries.
   - Keep global macro-F1 for model ranking, but display subject class coverage in LOSO drilldowns.

## Next Experiment Recommendation

Build v4 as a new model key, not a replacement:

```text
improved_gnn_lstm_temporal_node
improved_gnn_lstm_temporal_node_attn_adj
```

Run first on:

```text
acc16_gyro_hr / LOSO
```

Only scale to all three feature sets after it beats:

```text
v1 acc16_gyro_hr improved_gnn_lstm_attn_adj macro_f1 = 0.885490
```

