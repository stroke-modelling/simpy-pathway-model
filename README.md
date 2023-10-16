# simpy-pathway-model

Simpy pathway model for stroke thrombolysis and thrombectomy. This models compares the outcome difference between two models of care:

1. Patients attend their closest unit first for thrombolysis, with onward transfer for thrombectomy as required.
2. An assessment is done on scene and decision is made whether to bypass a local thrombolysis-only centre and take the patient further to a combined thrombolysis/thrombectomy centre.

```mermaid
flowchart LR
    A[Stroke\nonset] --> |Delay| B(Call 999)
    B --> |Delay| C(Ambulance\ndispatch + travel)
    C --> D(Ambulance\non-scene)
    D --> E{Decision}
    E --> |Travel| F(IVT-only Unit)
    E --> |Travel| G(IVT/MT Unit)
    F .-> |Transfer\nif required| G
    G --> H[Outcome]
    F --> H
```

