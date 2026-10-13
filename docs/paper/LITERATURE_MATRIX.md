# Literature matrix

| Cluster | Representative work | Relation to ERRORFlow | Direct score comparison? |
|---|---|---|---|
| Fact verification | FEVER; AVeriTeC; SciFact; CLIMATE-FEVER | Dataset/task foundations | Only with matched split, evidence, labels, metric |
| Robust verification | VitaminC; ProoFVer | Evidence sensitivity and structured entailment | No, different training/protocol |
| Self-correction | Self-Refine; RARR; CoVe | Frozen or iterative second-pass correction | No, different tasks and always-on protocols |
| Negative self-correction | Huang et al. | Motivates external outcome evidence | Conceptual |
| Selective computation | Selective QA; Active Retrieval; Adaptive-RAG; Self-RAG | Conditional compute/routing | Conceptual; several train routers |
| Non-parametric memory | RAG; CBR; Reflexion | Retrieve and reuse prior information | Conceptual; stored objects differ |
| Retrieval robustness | BEIR; Lost in the Middle; adaptive adversarial training | Domain shift, evidence ordering/noise | Diagnostic context |

ERRORFlow's narrow distinction is the combination of frozen correction, memory of observed helpful/harmful action outcomes, per-example routing, matched-call controls, and cross-ontology fact-verification evaluation.
