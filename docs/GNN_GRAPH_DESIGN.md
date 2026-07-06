# GNN Graph Design

## Blunt Assessment

The old 3-node PAMAP2 graph was possible and defensible, but too weak for the
claim we want to make. It compressed each body location into one statistical
summary node, so the GNN had very little channel/modality structure to reason
over. It was not literally a CNN, but it could behave like a small neural model
over engineered body-location summaries rather than a rich graph model.

The current default is therefore a hybrid graph.

## Current PAMAP2 Hybrid Graph

The PAMAP2 graph now contains both body-location aggregate nodes and raw-channel
nodes.

Body-location aggregate nodes:

- node 0: hand/wrist
- node 1: chest
- node 2: ankle

Channel nodes:

- one node per selected sensor channel at each body location
- examples: hand acc16 x, chest gyro z, ankle acc16 y

Optional global node:

- heart rate, when the selected feature set includes HR

The graph therefore has:

- `acc16_hr`: 3 location nodes + 9 channel nodes + 1 HR node = 13 nodes
- `acc16_gyro`: 3 location nodes + 18 channel nodes = 21 nodes
- `acc16_gyro_hr`: 3 location nodes + 18 channel nodes + 1 HR node = 22 nodes

Each node starts with 6 statistical descriptors:

- mean
- standard deviation
- minimum
- maximum
- RMS
- IQR

Each node also receives categorical context:

- node type: location, channel, or global
- body location: hand, chest, ankle, or global
- modality: aggregate, acc16, gyro, HR, or other
- axis: aggregate, x, y, z, or none

This gives a fixed PAMAP2 hybrid node feature dimension of 23:

- 6 statistical features
- 17 categorical context features

The graph tensor shapes now become:

- single-window GNN: `(batch, n_hybrid_nodes, 23)`
- sequence GNN-LSTM: `(batch, sequence_length, n_hybrid_nodes, 23)`

## What the GNN Learns

For a fixed-adjacency GNN:

1. Location nodes summarize body regions.
2. Channel nodes preserve modality/axis-level information.
3. Membership edges connect each channel to its body location.
4. Body-topology edges connect hand, chest, and ankle aggregate nodes.
5. Same modality/axis edges connect matching channels across body locations.
6. The optional HR node connects to body-location aggregate nodes.
7. The adjacency is normalized as `D^-1/2 A D^-1/2`.

Then:

1. The adjacency mixes information across this hybrid graph.
3. A linear layer transforms the mixed node features.
4. Nonlinearity and normalization produce node embeddings.
5. The model either pools/concatenates those node embeddings for classification,
   or passes per-window graph embeddings through an LSTM for temporal modeling.

For learnable/attention adjacency variants, the model can adjust edge strengths
instead of relying only on the fixed body-location graph.

## Legacy 3-Node Graph

The previous default used only one node per body location:

- hand/wrist
- chest
- ankle

That older graph asked:

> How do body locations interact during an activity?

The hybrid graph now asks:

> How do body locations, modalities, and axes interact during an activity?

If we want a strict ablation, we can keep the old 3-node graph as
`body_location_only`, but it should no longer be the primary GNN evidence.
