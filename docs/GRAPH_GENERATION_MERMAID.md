# Graph Generation Diagrams

These Mermaid diagrams describe the graph construction currently intended for
the canonical experiments.

## PAMAP2 Hybrid Graph Generation

```mermaid
flowchart TD
    A["Raw PAMAP2 window<br/>(T=128, C selected channels)"] --> B{"Feature set"}

    B -->|"acc16_hr"| C1["Channels:<br/>3 locations x acc16 xyz + HR<br/>C=10"]
    B -->|"acc16_gyro"| C2["Channels:<br/>3 locations x (acc16 xyz + gyro xyz)<br/>C=18"]
    B -->|"acc16_gyro_hr"| C3["Channels:<br/>3 locations x (acc16 xyz + gyro xyz) + HR<br/>C=19"]

    C1 --> D["Split by body location"]
    C2 --> D
    C3 --> D

    D --> E0["Location aggregate nodes<br/>hand, chest, ankle"]
    D --> E1["Per-channel nodes<br/>location x modality x axis"]
    D --> E2{"HR present?"}
    E2 -->|"yes"| E3["Global HR node"]
    E2 -->|"no"| E4["No HR node"]

    E0 --> F["Compute node features"]
    E1 --> F
    E3 --> F
    E4 --> F

    F --> F1["6 signal stats:<br/>mean, std, min, max, RMS, IQR"]
    F --> F2["Categorical context:<br/>node type, location, modality, axis"]
    F1 --> G["Node feature matrix<br/>N nodes x 23 features"]
    F2 --> G

    G --> H["Build hybrid adjacency"]
    H --> H1["Body topology edges:<br/>hand-chest, chest-ankle, hand-ankle"]
    H --> H2["Membership edges:<br/>location node to its channel nodes"]
    H --> H3["Within-location channel edges"]
    H --> H4["Same modality/axis edges<br/>across body locations"]
    H --> H5["HR-to-location edges<br/>if HR exists"]

    H1 --> I["Add self-loops"]
    H2 --> I
    H3 --> I
    H4 --> I
    H5 --> I
    I --> J["Normalize adjacency<br/>D^-1/2 A D^-1/2"]

    G --> K["GNN input"]
    J --> K
    K --> L["Single-window GNN:<br/>(batch, N, 23)"]
    K --> M["GNN-LSTM:<br/>(batch, seq_len, N, 23)"]
```

### PAMAP2 Node Counts

```mermaid
flowchart LR
    A["acc16_hr"] --> A1["3 location nodes"]
    A --> A2["9 channel nodes"]
    A --> A3["1 HR node"]
    A1 --> A4["13 total nodes"]
    A2 --> A4
    A3 --> A4

    B["acc16_gyro"] --> B1["3 location nodes"]
    B --> B2["18 channel nodes"]
    B1 --> B3["21 total nodes"]
    B2 --> B3

    C["acc16_gyro_hr"] --> C1["3 location nodes"]
    C --> C2["18 channel nodes"]
    C --> C3["1 HR node"]
    C1 --> C4["22 total nodes"]
    C2 --> C4
    C3 --> C4
```

## HHAR Graph Generation

The current HHAR graph is simpler because the processed HHAR path currently
uses accelerometer axes as graph nodes. If gyro is retained in a later canonical
HHAR feature path, this should be upgraded to the same hybrid location/channel
idea used for PAMAP2.

```mermaid
flowchart TD
    A["Raw HHAR window<br/>(T=128, selected channels)"] --> B["Select accelerometer axes<br/>x, y, z"]
    B --> C["Create axis nodes"]

    C --> C1["Node 0: accel x"]
    C --> C2["Node 1: accel y"]
    C --> C3["Node 2: accel z"]

    C1 --> D["Compute node features"]
    C2 --> D
    C3 --> D

    D --> D1["6 signal stats per axis:<br/>mean, std, min, max, RMS, IQR"]
    D1 --> E["Node feature matrix<br/>3 nodes x 6 features"]

    E --> F["Build fully connected axis adjacency"]
    F --> F1["x-y edge"]
    F --> F2["y-z edge"]
    F --> F3["x-z edge"]
    F1 --> G["Add self-loops"]
    F2 --> G
    F3 --> G
    G --> H["Normalize adjacency<br/>D^-1/2 A D^-1/2"]

    E --> I["GNN input"]
    H --> I
    I --> J["Single-window GNN:<br/>(batch, 3, 6)"]
    I --> K["GNN-LSTM:<br/>(batch, seq_len, 3, 6)"]
```

## Future HHAR Hybrid Option

```mermaid
flowchart TD
    A["HHAR phone/watch accel + gyro window"] --> B["Create device/location nodes<br/>phone, watch"]
    A --> C["Create channel nodes<br/>device x modality x axis"]
    B --> D["Membership edges<br/>device to channel"]
    C --> D
    C --> E["Same modality/axis edges<br/>phone accel x to watch accel x, etc."]
    B --> F["Device relation edge<br/>phone-watch"]
    D --> G["Hybrid HHAR graph"]
    E --> G
    F --> G
```
