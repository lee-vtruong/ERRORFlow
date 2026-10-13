# Citation verification

All bibliography entries were checked against primary venue pages: ACL Anthology, ICLR/OpenReview, NeurIPS proceedings, or arXiv/official institutional pages. `paper/scripts/check_citations.py` enforces unique keys and complete citation coverage.

Direct numerical comparison requires the same split, label space, evidence regime, and metric. FEVER, AVeriTeC shared-task, and SciFact systems are therefore contextual references rather than controlled baselines for ERRORFlow. Self-Refine, RARR, CoVe, selective QA, Adaptive-RAG, Self-RAG, case-based reasoning, and RAG are conceptual comparators.

The bibliography intentionally avoids unverified leaderboard values. CLIMATE-FEVER metadata follows its paper/arXiv and official UZH record; CRITIC was not included because its current official record does not provide stable proceedings metadata needed here.
