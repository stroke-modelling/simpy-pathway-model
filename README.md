# simpy-pathway-model
Simpy pathway model for stroke thrombolysis and thrombectomy

```mermaid
flowchart LR
    A[fa:fa-home Stroke\nonset] --> |Delay| B(fa:fa-phone Call 999)
    B --> |Delay| C(fa:fa-ambulance Ambulance\ndispatch + travel)
    C --> D(fa:fa-ambulance Ambulance\non-scene)
    D --> E{fa:fa-user-md\nDecision}
    E --> |Travel| F(fa:fa-syringe IVT-only Unit)
    E --> |Travel| G(fa:fa-procedures IVT/MT Unit)
    F .-> |Transfer\nif required| G
    G --> H[fa:fa-walking Outcome]
    F --> H
```

