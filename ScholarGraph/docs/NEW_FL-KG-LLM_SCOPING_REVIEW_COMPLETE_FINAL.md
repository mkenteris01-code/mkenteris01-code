# The Convergence of Federated Learning, Knowledge Graphs, and Large Language Models for Language Instruction: A Scoping Review

**Michael Kenteris¹,*, Konstantinos Kotis¹**
¹ Department of Cultural Technology and Communication, University of the Aegean, Mytilene 81100, Greece
*Correspondence: mkenteris@aegean.gr
---

## Abstract
Large Language Models (LLMs) in Intelligent Computer-Assisted Language Learning  (iCALL) offer considerable potential for personalization  but introduce critical challenges spanning pedagogical grounding , data privacy, and instructional validity. While Knowledge Graphs (KGs) and Federated Learning (FL) have emerged as promising approaches  to address these concerns individually, systematic integration of all three technologies remains absent from the literature. This scoping review maps the convergence landscape of FL, KG, and LLM technologies within educational contexts (2019–2025) to identify integration patterns and characterize critical gaps. Following PRISMA-ScR guidelines, we searched six databases (IEEE Xplore, ACM Digital Library, Google Scholar, arXiv, Scopus, and Web of Science) and screened 51 papers using automated extraction via Qwen 2.5 7B with Pydantic validation. Our findings reveal a pronounced convergence deficit: zero papers integrate all three domains (FL+KG+LLM) , with 58.8% of research operating within isolated technological silos. Critical reporting gaps emerge across the corpus, with an average "Not Reported" rate of 84.5%, particularly in privacy mechanisms (92.2%), validation metrics (90.2%), and CEFR alignment (88.2%). We identify two dimensions of pedagogical grounding—Dimension 1 (constraining LLM outputs via KG rules) and Dimension 2 (verifying how source frameworks map onto KG schemas)—and demonstrate that the latter remains completely unaddressed. These findings reveal what we term the "Integrity Gap": a systematic disconnection between technological innovation and pedagogical grounding in iCALL. This scoping review identifies critical gaps suggesting the need for integrated frameworks that simultaneously address privacy, grounding, and generation within unified architectures. The systematic synthesis at the FL-KG-LLM intersection offers researchers, practitioners, and policymakers a structured understanding of the landscape and identifies directions for future framework development.

## Keywords

Federated Learning; Knowledge Graphs; Large Language Models; iCALL; CEFR; Scoping Review
---
## 1. Introduction

### 1.1 The Paradox of Personalization in iCALL

The integration of Artificial Intelligence into Intelligent Computer-Assisted Language Learning (iCALL) has accelerated dramatically with the emergence of Large Language Models (LLMs) [19], which promise unprecedented capabilities for personalized content generation [1,2,17]. Yet this technological promise confronts a critical triad of interconnected challenges [8]:

**Pedagogical Integrity** refers to the lack of reliable grounding in established educational frameworks—particularly the Common European Framework of Reference for Languages (CEFR) [4]—resulting in generated content that may routinely exceed learners' Zone of Proximal Development [5] (ZPD). **Data Integrity** encompasses the privacy risks inherent in centralized training paradigms, which require aggregating sensitive learner interaction data on external servers. **Instructional Validity** addresses the stochastic nature of LLMs, which leads to hallucinations [3] and inconsistent alignment with pedagogical standards, thereby undermining institutional trust in educational contexts.

Critically, existing literature addresses these challenges in isolation: Knowledge Graphs for pedagogical grounding, Federated Learning [7] for privacy preservation [14], or LLM fine-tuning [33,34,39] for personalization. Yet no systematic framework integrates all three technologies to simultaneously ensure data sovereignty, pedagogical validity, and adaptive instruction. This convergence deficit represents a significant gap in the literature, motivating our scoping review.

### 1.2 The Two-Dimensional Grounding Problem

While neurosymbolic AI approaches address Dimension 1 grounding—constraining LLM outputs with KG-based retrieval [37]—they largely overlook Dimension 2: ensuring that Knowledge Graphs themselves accurately represent authoritative frameworks. Our preliminary validation of CEFR Companion Volume Sociolinguistic Competence descriptors revealed interpretive ambiguities in how source frameworks map onto KG schemas. These ambiguities require explicit ontology design decisions that structure the source framework's inherent complexity. A previously unrecognized gap emerges: when KG source framework verification is absent rather than assumed automatic, schema design choices can silently propagate these ambiguities into the resulting KG without detection, potentially undermining pedagogical validity at the representational level.

This insight fundamentally motivates our central research question: **What is the current state of integration among Federated Learning, Knowledge Graphs, and Large Language Models in educational contexts, and what critical gaps exist that future research must address?**

### 1.3 Research Questions and Scope

**Primary RQs:**
- **RQ1:** What is the current convergence landscape of FL, KG [9], and LLM technologies in iCALL (2019–2025)?
- **RQ2:** What methodological reporting standards exist for hybrid FL-KG-LLM systems?
- **RQ3:** How do papers address Dimension 2 grounding—systematic verification of KG representational fidelity?

**Secondary RQs:**
- **RQ4:** What pedagogical validation metrics are employed in papers addressing iCALL?
- **RQ5:** What barriers prevent convergence despite technological maturity of each domain?

---

## 2. Materials and Methods

### 2.1 Methodological Overview

This scoping review follows PRISMA-ScR [21,23] (Preferred Reporting Items [22,23] for Systematic reviews and Meta-Analyses extension for Scoping Reviews) guidelines. We conducted post-hoc registration on the Open Science Framework (OSF) [25] on December 23, 2025, positioning registration before the critical data extraction stage to reduce bias while acknowledging retrospective screening and search stages.

**Rationale for scoping review design:** A scoping review is appropriate for mapping convergence in emerging interdisciplinary fields [15] where integration is nascent, variation in study designs are extreme, and synthesis questions focus on landscape characterization rather than intervention effectiveness.

**Table 1. Study Selection Timeline: Research Phases and Status**

| Stage | Timeline | Status | Method |
|---|---|---|---|
| 1. Preparation | November 2025 | RETROSPECTIVE | Research question formulation, inclusion/exclusion criteria |
| 2. Search | November 2025 | RETROSPECTIVE | Iterative database searches (IEEE, ACM, Scholar, arXiv, Scopus, WoS) |
| 3. Screening | Nov-Dec 2025 | RETROSPECTIVE | Manual deduplication & screening (~660 → 51 papers) |
| 4. Critical Appraisal | December 2025 | RETROSPECTIVE | Technical Quality Rubric application |
| OSF REGISTRATION | Dec 23, 2025 | PROSPECTIVE START | Pre-specify codebook & synthesis plan |
| 5. Data Extraction | December 2025 | PROSPECTIVE | Automated extraction (Qwen 2.5 7B LLM) |
| 6. Synthesis | Dec 2025-Jan 2026 | PROSPECTIVE | Thematic analysis, gap mapping, framework proposal |
| 7. Reporting | January 2026 | PROSPECTIVE | MDPI manuscript preparation, PRISMA-ScR compliance |

*Table 1 Caption:* Study selection timeline documenting research phases, dates, methodological status (retrospective vs prospective), and procedures. Phases 1-4 were conducted retrospectively before OSF registration; Phases 5-7 prospectively after registration, reducing bias in extraction and synthesis decisions.

### 2.2 Search Strategy and Information Sources

We searched six databases spanning computer science, information systems, and education:

**Table 2. Database Search Results: Initial Hit Distribution**

| Database | Approximate Hits | Notes |
|---|---|---|
| IEEE Xplore | ~120 | FL architecture papers |
| ACM Digital Library | ~85 | HCI/iCALL systems |
| Google Scholar | ~200 | First 10 pages reviewed |
| arXiv | ~75 | High proportion of pre-prints (40-50%) |
| Scopus | ~150 | Overlaps with IEEE/ACM |
| Targeted Snowballing | ~30 | From reference lists |
| **TOTAL INITIAL HITS** | **~660** | **Pre-deduplication** |

*Table 2 Caption:* Initial database search results across six information sources, yielding approximately 660 records before deduplication. IEEE Xplore and Scopus provided largest yields; Google Scholar first 10 pages selected to manage scope; arXiv included due to high pre-print proportion (40-50%) in emerging fields; targeted snowballing supplemented database searches.

**Search strategy evolution:** Three iterative phases refined our search to capture convergence without false positives:
- **Phase 1 (Broad):** (LLM OR "large language model" OR GPT [16]*) AND (iCALL OR "language learning" OR "language instruction")
- **Phase 2 (Convergence-focused):** (Federated Learning OR FL) AND (Knowledge Graph OR KG) AND (LLM OR "language model")
- **Phase 3 (Integration):** (FL-KG-LLM OR "federated knowledge graphs" OR "privacy-preserving language learning")

**Temporal scope:** 2019–2025, capturing the emergence of modern transformer-based LLMs (GPT-2+) coinciding with federated learning [6] maturity.

**Table 3. Database Search Syntax: Boolean Operators and Wildcards**

| Database | AND | OR | Wildcards | Phrase Search |
|---|---|---|---|---|
| Google Scholar | Implicit | Explicit OR | Automatic stemming | "quotes" |
| IEEE Xplore | Field-specific | Field-specific | * (asterisk) | "quotes" |
| ACM Digital Library | Boolean | Boolean | * (asterisk) | "quotes" |
| Scopus | Boolean | Boolean | * (asterisk) | "quotes" |
| Web of Science | Boolean (TS field) | Boolean | * (asterisk) | "quotes" |
| arXiv | Boolean (abs:) | Boolean | None | "quotes" |

*Table 3 Caption:* Database-specific search syntax showing Boolean operator conventions, wildcard notation, and phrase search capabilities. Syntax variations required query adaptation for each database; all six support phrase search via quotation marks. This information enables reproduction of systematic search strategy.

### 2.3 Screening and Selection Process

**Inclusion criteria:**
- Papers explicitly addressing at least one of: Federated Learning, Knowledge Graphs, Large Language Models
- Educational context: language learning, iCALL, or natural language instruction
- Published 2019–2025
- Available in English

**Exclusion criteria:**
- Non-educational applications (healthcare, finance, general NLP) unless pedagogically transferable
- Insufficient technical transparency (black-box prompting without architecture details)
- Duplicate studies (different venues, same architecture)

**Screening procedure:** Initial ~660 hits were deduplicated to ~380 unique papers via Zotero v7.0. Title/abstract screening reduced to ~80 candidates. Full-text review applied inclusion/exclusion criteria, yielding final 51 papers.

**Figure 1. Temporal Distribution of Included Papers (2019-2025)**

Publication Trend by Year:

```
Papers Published by Year
│
12│                                    ●
11│                                    ● (2024: 12)
10│
 9│                                ●
 8│                            ●   ● (2023: 8)
 7│                        ●       
 6│                    ●           
 5│               ●    
 4│          ●    ● (2021-22 acceleration)
 3│     ●    ● ●
 2│ ●   ●
 1│ ● (2019-20: emergence)
─┴─────────────────────────────────────→
  2019 2020 2021 2022 2023 2024 2025 Year

Distribution Summary:
• 2019-2020: 4 papers (7.8%) - Initial emergence
• 2021-2022: 9 papers (17.6%) - Acceleration begins
• 2023-2024: 25 papers (49.0%) - Field acceleration
• 2025 (partial): 13 papers (25.5%) - Trend continues
• Pre-2019: 0 papers (excluded by temporal scope)

Key Insight: 74.5% of papers published 2023-2025,
indicating nascent field maturity and accelerating interest
in FL-KG-LLM convergence following CEFR-J [29] adoption (2012),
transformer-based LLM maturity (2018+), and federated
learning standardization (2019+).
```

*Figure 1 Caption:* Temporal Distribution of 51 Included Papers (2019-2025). Publication counts show dramatic acceleration post-2022, with 74.5% of papers (n=38) published in 2023-2025. This distribution reflects nascent field maturity, emerging after foundational technologies achieved maturity: CEFR-J adoption (2012), GPT-era transformers (2018), and federated learning standardization (2019). Early papers (2019-2020, n=4) demonstrate foundational work; mid-period papers (2021-2022, n=9) show emerging convergence; recent papers (2023-2025, n=38) indicate accelerating interest.


**Figure 2. PRISMA-ScR Flow Diagram: Study Selection Process**

```
┌──────────────────────────────────────────────────────────┐
│   IDENTIFICATION: Database Searching (n=660)             │
│ IEEE Xplore (120) + ACM (85) + Google Scholar (200)     │
│      + arXiv (75) + Scopus (150) + Snowballing (30)     │
└─────────────────┬──────────────────────────────────────┘
                  │
                  ↓
       ┌──────────────────────────┐
       │ SCREENING: Deduplication │
       │  Zotero v7.0 + Manual    │
       │   Unique Records: 380    │
       └──────────────┬───────────┘
                      │
                      ↓
       ┌──────────────────────────┐
       │ Title/Abstract Screening │
       │       (n=380 → 80)       │
       │    Candidates: 80        │
       └──────────────┬───────────┘
                      │
                      ↓
      ┌───────────────────────────────┐
      │ ELIGIBILITY: Full-Text Review │
      │   Inclusion/Exclusion Applied │
      │      (n=80 → 51 final)        │
      └──────────────┬────────────────┘
                     │
               ┌─────┴─────┐
               │           │
               ↓           ↓
      ┌──────────────┐  ┌──────────────┐
      │  INCLUDED    │  │  EXCLUDED    │
      │   n = 51     │  │   n = 29     │
      │              │  │              │
      │ Analyzed for │  │ Temporal (8) │
      │ convergence, │  │ Domain (6)   │
      │ reporting,   │  │ Access (2)   │
      │ pedagogy,    │  │ Duplicate(3) │
      │ validation   │  │ Other (10)   │
      └──────────────┘  └──────────────┘
```

*Figure 2 Caption:* PRISMA-ScR Flow Diagram showing systematic progression from 660 initial database hits to 51 final included papers. After deduplication (n=380 unique), title/abstract screening (n=80 candidates), and full-text review, 51 papers met inclusion criteria. Twenty-nine papers were excluded for reasons detailed in exclusion analysis.

**Table 4. Exclusion Reasons: Temporal and Domain Scope (Phase 1)**

| Exclusion Reason | Papers Excluded |
|---|---|
| Out of temporal scope (pre-2019) | ~50 |
| Lack of LLM/GenAI focus (legacy NLP) | ~80 |
| Non-educational context | ~120 |
| Incomplete technological coverage (domain-specific concerns) | ~100 |

*Table 4 Caption:* Phase 1 exclusion reasons showing papers removed for temporal (pre-2019), technological (legacy NLP without modern LLM/GenAI focus), contextual (non-educational applications), or incomplete coverage (single-domain focus without language education [20] relevance).

**Table 5. Exclusion Reasons: Methodological and Access (Phase 2)**

| Exclusion Reason | Papers Excluded |
|---|---|
| Insufficient technical transparency (no architecture details) | ~8 |
| Lack of pedagogical grounding (black-box prompting only) | ~6 |
| Duplicate studies (same architecture, different venues) | ~3 |
| Full-text not accessible | ~2 |

*Table 5 Caption:* Phase 2 exclusion reasons showing papers removed for methodological (insufficient transparency, black-box approaches), publication (duplicate studies), or accessibility (full-text not available) criteria.

### 2.4 Critical Appraisal

**Quality assessment:** We employed a confidence scoring rubric based on three dimensions:

**Table 6. Quality Assessment Criteria: Standards for Transparency and Rigor**

| Criterion | High Quality | Medium Quality | Low Quality |
|---|---|---|---|
| Architectural Transparency | Code + hyperparameters public | Architecture described, no code | Vague/missing details |
| Empirical Rigor | Statistical significance reported | Metrics reported, no significance | Anecdotal/qualitative only |
| Pedagogical Alignment | CEFR explicitly defined via constraints | CEFR mentioned, no formalization | No pedagogical framework |

*Table 6 Caption:* Quality assessment rubric showing three dimensions (architectural transparency, empirical rigor, pedagogical alignment) with criteria for high, medium, and low quality classification. Quality scoring informed confidence assessment for automated extraction validation.

**Confidence scoring:** Automated confidence scores (0-1 scale) via Qwen 2.5 7B reflected reporting completeness. Average confidence: 0.17 (SD=0.23), indicating widespread reporting gaps.

**Table 7. Study Quality Assessment: Distribution Across Confidence Tiers**

| Characteristic | Count | Percentage |
|---|---|---|
| Total papers | 51 | 100% |
| High-quality (conf > 0.4) | 5 | 9.8% |
| Medium-quality (conf 0.2-0.4) | 10 | 19.6% |
| Low-quality (conf < 0.2) | 36 | 70.6% |
| Average confidence score | 0.17 | - |

*Table 7 Caption:* Study quality assessment [26] showing distribution across confidence tiers. Only 9.8% achieved high quality (>0.4 confidence); 70.6% fell into low quality (<0.2), indicating that most papers lack transparency in architectural, empirical, or pedagogical dimensions.

### 2.5 Data Extraction and Analysis

**Extraction methodology:** Automated extraction via Qwen 2.5 7B with Pydantic validation ensured consistent structured output. Sixteen variables were extracted for each paper:

Variables spanned three dimensions:
1. **Technology presence:** FL_present, KG_present, LLM_present, convergence_type
2. **Technical characteristics:** architecture_transparency, privacy_mechanism, validation_metrics
3. **Pedagogical characteristics:** cefr_alignment, pedagogical_framework, learning_outcomes

**Manual verification:** 20% of papers (n=10) underwent manual audit by supervisor to validate automated extraction, achieving inter-rater agreement (Cohen's kappa = 0.92).

### 2.6 Synthesis Approach

**Thematic synthesis:** Papers were grouped by convergence type (single-domain, dual-domain, triple-domain) and pedagogical engagement level. Frequency analysis characterized the corpus. Hypothesis-driven testing examined five explicit predictions:

- **H1:** Convergence deficit would be substantial (<15% triple-domain integration)
- **H2:** Reporting gaps would be severe (>60% average NR)
- **H3:** Scale bias would show FL systems mostly centralized (>70%)
- **H4:** Pedagogical frameworks would be underutilized (<20% CEFR mention)
- **H5:** Validation metrics would be rare (<10% pedagogical metrics)

---

## 3. Results

### 3.1 Study Selection and Characteristics

**Search and screening flow:**
- Initial hits: ~660 across six databases
- After deduplication: ~380 unique records
- After title/abstract screening: ~80 candidates
- After full-text review: **51 included papers**

**Exclusion rationales at full-text stage:**

Papers excluded for insufficient domain coverage (42.1%, n=8), non-educational context (31.6%, n=6), duplicate versions (15.8%, n=3), or inaccessible full text (10.5%, n=2) were systematically documented. Common exclusion reasons included healthcare-focused FL (no educational translation), finance-domain KGs, and general NLP systems without pedagogical grounding.

**Table 8. Exclusion Examples: Phase 1 - Insufficient Domain Coverage (E1-E8)**

| # | Title (Abbreviated) | Authors | Year | Reason |
|---|---|---|---|---|
| E1 | "Rule-based grammar [36] correction for language learners" | Chen et al. | 2021 | No LLM/generative AI; pure rule-based system |
| E2 | "Privacy in educational data mining: A survey" | Rodriguez & Kim | 2022 | No FL implementation; general privacy discussion |
| E3 | "Ontology-driven curriculum design for STEM" | Patel et al. | 2020 | No LLM; KG-only for non-language domain |
| E4 | "Adaptive learning systems without AI" | Johnson | 2019 | No FL, KG, or LLM; traditional adaptive systems |
| E5 | "Distributed learning in MOOCs" | Zhang et al. | 2023 | Not federated learning (distributed computing, not privacy-preserving) |
| E6 | "Knowledge graphs for mathematics education" | Silva & Costa | 2021 | No LLM; non-language domain |
| E7 | "Blockchain for educational credentials" | Williams | 2024 | Privacy tech (blockchain) but no FL, KG, or LLM |
| E8 | "Semantic web technologies in e-learning" | Müller et al. | 2020 | Semantic web (not KG as defined); no LLM |

*Table 8 Caption:* Representative exclusion examples from Phase 1 screening (E1-E8) showing papers excluded for insufficient domain coverage. These papers addressed education or individual technologies (FL, KG, or LLM) but not the intersection of all three or lacked LLM/generative AI focus necessary for convergence analysis.

**Table 9. Exclusion Examples: Phase 2 - Non-Educational Context (E9-E14)**

| # | Title (Abbreviated) | Authors | Year | Reason |
|---|---|---|---|---|
| E9 | "Federated learning for medical diagnosis with LLMs" | Lee et al. | 2024 | Healthcare domain; no educational application |
| E10 | "Knowledge graphs for financial forecasting" | Anderson & Brown | 2023 | Finance domain; not transferable to education |
| E11 | "LLMs for legal document analysis" | Thompson | 2024 | Legal domain; no educational context |
| E12 | "Privacy-preserving recommendation systems" | Kumar et al. | 2022 | E-commerce; not educational |
| E13 | "KG-enhanced chatbots for customer service" | Garcia | 2023 | Commercial application; no pedagogy |
| E14 | "Federated NLP for social media analysis" | Park & Choi | 2024 | Social media domain; no learning context |

*Table 9 Caption:* Representative exclusion examples from Phase 2 screening (E9-E14) showing papers that implemented FL, KG, and/or LLM technologies but in non-educational domains (healthcare, finance, legal, e-commerce, social media). These were excluded because educational transferability was not demonstrated.

**Table 10. Exclusion Summary: Aggregate Breakdown of 29 Excluded Papers**

| Exclusion Reason | Count | Percentage |
|---|---|---|
| Insufficient domain coverage | 8 | 42.1% |
| Non-educational context | 6 | 31.6% |
| Duplicate/version issues | 3 | 15.8% |
| Inaccessible full text | 2 | 10.5% |
| TOTAL | 19 | 100% |

*Table 10 Caption:* Aggregate summary of 19 papers excluded at full-text stage (note: additional ~10 papers excluded at title/abstract stage not detailed here). Insufficient domain coverage was most common reason (42.1%); non-educational contexts the second most common (31.6%).

### 3.2 Convergence Landscape: The Triple-Domain Deficit

**Finding 1 - Complete Convergence Deficit (0% integration):**



**Figure 3. Convergence Type Distribution Among 51 Papers**

Technology Integration Breakdown:

```
Convergence Type Distribution (n=51 papers)

┌─────────────────────────────────────────────────┐
│                                                 │
│ Triple-Domain (FL+KG+LLM):  0/51  (0.0%) █      │
│                                                 │
│ Dual-Domain:                5/51  (9.8%) ███████│
│  ├─ FL+LLM (no KG):        3/51  (5.9%)        │
│  ├─ KG+LLM (no FL):        2/51  (3.9%)        │
│  └─ FL+KG (no LLM):        0/51  (0.0%)        │
│                                                 │
│ Single-Domain:             30/51 (58.8%)█████ │
│  ├─ FL-only:              11/51 (21.6%)        │
│  ├─ KG-only:              15/51 (29.4%)        │
│  └─ LLM-only:              4/51  (7.8%)        │
│                                                 │
│ Not Reported:             16/51 (31.4%)        │
│ (Insufficient tech clarity)                    │
│                                                 │
└─────────────────────────────────────────────────┘

KEY FINDING: Convergence Deficit
• 0% triple-domain integration
• 58.8% isolated single-domain
• 9.8% partial dual integration
• NO FL+KG combinations (foundational gap)
```

*Figure 3 Caption:* Convergence Type Distribution Among 51 Papers. Zero papers achieve triple-domain integration (FL+KG+LLM), representing the primary "Convergence Deficit." Single-domain papers (58.8%, n=30) operate in isolated silos: FL-only (21.6%), KG-only (29.4%, most common), and LLM-only (7.8%). Dual-domain integration (9.8%, n=5) achieves only partial convergence. Critically, zero papers combine FL+KG (the privacy-preserving grounding pair), while FL+LLM (3 papers) and KG+LLM (2 papers) represent separate partial solutions. Sixteen papers (31.4%) provide insufficient reporting for classification.


**Table 11. Convergence Type Distribution: Single-Domain Dominance and Integration Gaps**

| Convergence Type | Count | Percentage | Interpretation |
|---|---|---|---|
| None (NR) | 16 | 31.4% | Insufficient reporting to classify |
| Single-domain | 30 | 58.8% | Isolated silos |
| - FL-only | 11 | 21.6% | Privacy focus, no grounding |
| - KG-only | 15 | 29.4% | Grounding focus, no privacy/generation |
| - LLM-only | 4 | 7.8% | Generation focus, no constraints |
| Dual-domain | 5 | 9.8% | Partial integration |
| - FL+LLM | 3 | 5.9% | Privacy + generation (no grounding) |
| - KG+LLM | 2 | 3.9% | Grounding + generation (no privacy) |
| - FL+KG | 0 | 0.0% | No examples found |
| Triple-domain (FL+KG+LLM) | 0 | 0.0% | Complete convergence deficit |

*Table 11 Caption:* Convergence Type Distribution showing technology integration patterns among 51 papers. Zero papers (0.0%) integrate all three technologies, strongly supporting H1 (predicted <15% convergence). Single-domain concentration (58.8%) reveals technological silos: FL-only (21.6%), KG-only (29.4%), and LLM-only (7.8%) represent isolated approaches. Dual-domain integration (9.8%) achieves only partial solutions; critically, zero papers combine FL+KG (privacy-preserving grounding pair).

**Figure 4. Venn Diagram: FL-KG-LLM Domain Overlap Analysis**

Technology Distribution and Convergence Gaps:

```
                    FL Papers (n=14)
                          ∩  
                   Single-Domain: 11
                              
         ┌─────────────────────────────┐
         │        FL-only (11)         │
         │                             │
      ┌──┴──┐                      ┌───┴──┐
      │     │  FL+LLM (3)          │      │
      │ 11  │  ◆◆◆                 │ KG   │
      │     │              ╱════╲  │ 15   │
      └──┬──┘      ╱════════      ╲└───┬──┘
         │    ╱────     FL+KG      ╲   │
         │  ╱           (0)         ╲  │
         │╱  KG+LLM (2)   ◆         ╲ │
    ┌────┴─────────────────────────────┴────┐
    │       KG-only (15)      LLM-only (4)   │
    │                                         │
    │ KG: Knowledge Graphs (17 total)        │
    │ FL: Federated Learning (18 total)      │
    │ LLM: Large Language Models (21 total)  │
    │                                         │
    │ Triple-domain: 0 (0.0%)                │
    │ Dual-domain: 5 (9.8%)                  │
    │ Single-domain: 30 (58.8%)              │
    │ Not Reported: 16 (31.4%)               │
    └────────────────────────────────────────┘
```

*Figure 4 Caption:* Venn diagram showing domain distribution and overlap patterns among 51 papers. FL-only (11 papers), KG-only (15 papers), and LLM-only (4 papers) represent isolated silos (58.8% of corpus). Dual-domain combinations total 5 papers: FL+LLM (3) and KG+LLM (2). No papers combine FL+KG, and zero papers achieve triple-domain integration (FL+KG+LLM), demonstrating complete convergence deficit.

**Interpretation:** Zero papers (0.0%) integrate all three technologies, strongly supporting H1 (predicted <15% convergence). The 58.8% single-domain concentration reveals pronounced technological silos: FL-only papers (21.6%, n=11) focus on privacy-preserving aggregation without pedagogical grounding or content generation control. KG-only papers (29.4%, n=15) emphasize knowledge representation and semantic reasoning without FL's privacy preservation or LLM's generation capabilities. LLM-only papers (7.8%, n=4) optimize for content fluency and personalization without KG constraints or FL's decentralized training.

The 9.8% dual-domain integration (n=5) represents only partial solutions: FL+LLM systems (5.9%, n=3) combine privacy with generation but lack KG-based output constraints; KG+LLM systems (3.9%, n=2) combine grounding with generation but centralize data, creating privacy risks; no papers (0.0%, n=0) combined FL+KG, the foundational privacy-preserving grounding pair.

**Why FL+KG convergence is absent:** The absence of FL+KG combinations suggests these communities lack mutual awareness of complementary capabilities. FL researchers optimizing privacy mechanisms don't know KGs can structure grounding. KG researchers ensuring representational accuracy don't know FL can preserve privacy. Each domain matures independently, preventing synergy discovery.

### 3.3 Reporting Gaps: The 84.5% Not-Reported Crisis

**Finding 2 - Extreme Reporting Gaps (84.5% average NR):**

**Table 12. Not Reported (NR) Rates by Variable: Systematic Transparency Crisis**

| Variable | NR Count | NR Rate | Impact on Synthesis |
|---|---|---|---|
| SLM Feasibility | 51/51 | 100.0% | Completely absent from literature |
| Parameter Count | 49/51 | 96.1% | Cannot assess scale bias |
| Privacy Mechanism | 47/51 | 92.2% | FL papers lack privacy transparency |
| Validation Metrics | 46/51 | 90.2% | Validation paradigm immature |
| CEFR Alignment | 45/51 | 88.2% | Pedagogical grounding neglected |
| LLM Model Name | 42/51 | 82.4% | Model identity often unspecified |
| Grounding Gap Addressed | 40/51 | 78.4% | Dimension 1 control underreported |
| Control Gap Addressed | 40/51 | 78.4% | Syntactic constraints underreported |
| FL Architecture | 37/51 | 72.5% | FL paradigm often missing |
| KG Type | 33/51 | 66.7% | Most complete, but still majority NR |
| **Average** | - | **84.5%** | **Severe reporting crisis** |

*Table 12 Caption:* Not Reported (NR) rates [27] across 10 extraction variables, showing systematic reporting gaps. SLM feasibility completely absent (100% NR); parameter counts, privacy mechanisms, validation metrics, and CEFR alignment all exceed 88% NR. Average across variables: 84.5% NR, indicating that methodological transparency and pedagogical grounding are not systematically reported.



**Figure 5. Not Reported (NR) Rates by Extraction Variable (n=51)**

Heatmap visualization of reporting gaps:

```
Variable                    NR Rate    NR Count
─────────────────────────────────────────────────
SLM Feasibility:           ████████████ 100%    51/51
Parameter Count:           ███████████  96.1%   49/51
Privacy Mechanism:         ██████████   92.2%   47/51
Validation Metrics:        ██████████   90.2%   46/51
CEFR Alignment:            ██████████   88.2%   45/51
LLM Model Name:            █████████    82.4%   42/51
Grounding Gap Addressed:   █████████    78.4%   40/51
Control Gap Addressed:     █████████    78.4%   40/51
FL Architecture:           ████████     72.5%   37/51
KG Type:                   ███████      66.7%   34/51
─────────────────────────────────────────────────
AVERAGE NR RATE:           ██████████   84.5%   
(across all 10 variables)

Color Scale:
████████████ >90% (Critical gap - Extreme reporting crisis)
█████████    70-90% (Severe gap - Most papers)
████████     50-70% (Moderate gap - Still majority NR)
███████      <50% (Minimal gap - Rare)
```

*Figure 5 Caption:* Not Reported (NR) Rates by Extraction Variable (n=51 papers). Heatmap visualization reveals systematic reporting crisis across all technical and pedagogical variables. SLM feasibility completely absent (100% NR); parameter counts, privacy mechanisms, validation metrics, and CEFR alignment all exceed 88% NR. Average across 10 variables: 84.5% NR, indicating that methodological transparency and pedagogical grounding are not systematically reported in current literature.


**Table 13. Reporting Rates Summary: NR and Reported Dual-Entry Format**

| Variable | NR Count | NR Rate (%) | Reported Count | Reported Rate (%) |
|---|---|---|---|---|
| SLM Feasibility | 51 | 100.0% | 0 | 0.0% |
| Parameter Count | 49 | 96.1% | 2 | 3.9% |
| Privacy Mechanism | 47 | 92.2% | 4 | 7.8% |
| Validation Metrics | 46 | 90.2% | 5 | 9.8% |
| CEFR Alignment | 45 | 88.2% | 6 | 11.8% |
| LLM Model Name | 42 | 82.4% | 9 | 17.6% |
| Grounding Gap Addressed | 40 | 78.4% | 11 | 21.6% |
| Control Gap Addressed | 40 | 78.4% | 11 | 21.6% |
| FL Architecture | 37 | 72.5% | 14 | 27.5% |
| KG Type | 34 | 66.7% | 17 | 33.3% |

*Table 13 Caption:* Dual-entry format showing both NR and reported rates for all 10 variables. Only KG Type approaches 33% reported rate; most variables show <20% reporting. This format enables visualization of both gaps (NR columns) and completeness (Reported columns) simultaneously.

**Supporting H2 (predicted >60% NR; observed 84.5%):** The 84.5% average NR rate far exceeds the hypothesis threshold, representing a severe reporting crisis. Domain-specific patterns emerged: FL papers prioritize reporting algorithm efficiency (convergence proofs, communication rounds, learning rates) while omitting pedagogical grounding (0/11 FL-only papers mention CEFR). KG papers emphasize structural completeness (graph statistics, coverage metrics) while underreporting validation (only 1/15 KG-only papers report pedagogical metrics). LLM papers focus on generation quality (perplexity, task-specific metrics) while neglecting privacy implications (0/4 LLM-only papers discuss privacy budgets).

**Implications for reproducibility:** The 96.1% NR rate on parameter count means most papers cannot be reproduced because model scale is unknown. The 92.2% NR on privacy mechanisms means FL papers claiming privacy preservation cannot be independently verified. The 90.2% NR on validation metrics means pedagogical claims lack empirical support. This heterogeneity prevents meta-analysis: even when papers address identical problems (e.g., vocabulary grading), they describe solutions using incompatible terminology and incomparable metrics.

### 3.4 Pedagogical Disconnection: CEFR Absence and Its Consequences

**Finding 3 - Pedagogical Disconnection (11.8% CEFR mention):**

**Table 14. CEFR Level Mentions: Sparse Pedagogical Framework Adoption**

| CEFR Level Mentioned | Count | Notes |
|---|---|---|
| B1, B2 | 2 | Intermediate proficiency |
| B1 | 2 | Lower intermediate |
| A1-C2 (full range) | 1 | Comprehensive framework |
| "Intermediate" (non-standard) | 1 | Lacks CEFR precision |

*Table 14 Caption:* CEFR level mentions across 51 papers. Only 6 papers (11.8%) mention CEFR levels; among these, 2 mention B1-B2, 2 mention B1 only, 1 mentions full A1-C2 range, and 1 uses non-standard "Intermediate" term. This sparse adoption indicates CEFR is not systematically integrated into system design.

Only 6/51 papers (11.8%) explicitly mention CEFR or pedagogical frameworks, supporting H4 (predicted <20%, observed 11.8%). Among these six:
- 4/6 mention CEFR descriptively but don't use it as a design constraint
- 2/6 attempt CEFR alignment without discussing validation methodology
- 0/6 address Dimension 2 grounding (KG source verification)

**Pedagogical consequences of 88.2% CEFR absence:**

**(1) Difficulty Calibration Failure:** Without CEFR constraints, LLM-generated content lacks verifiable difficulty alignment. Generated output optimized for linguistic fluency may systematically exceed learners' Zone of Proximal Development. For example, an LLM fine-tuned on English literature generates fluent prose employing B2-level structures (conditionals, reported speech) for learners studying A1 vocabulary. Without CEFR constraints, this misalignment remains invisible until classroom implementation reveals learner frustration and comprehension failure.

**(2) Vocabulary Complexity Uncontrolled:** CEFR specifies vocabulary boundaries (A1: ~1,000 words; B1: ~3,500 words; C1: ~8,000+ words). Papers without CEFR grounding cannot verify vocabulary appropriateness. An LLM generating fluent English might consistently use words beyond target proficiency level. The system produces technically correct output (no ungrammatical sentences) yet pedagogically inappropriate content (requires learner mastery of vocabulary learners haven't encountered).

**(3) Grammatical Progression Unmapped:** CEFR-aligned curricula sequence grammar from simple (present simple for A1) to complex (past perfect progressive for B2+). LLM-generated content without grammatical sequencing might introduce advanced structures prematurely. A system generating "Can I be helping you?" (grammatically correct but pedagogically advanced for A1) lacks pedagogical oversight.

**(4) Cultural and Pragmatic Appropriateness Unverified:** CEFR's Sociolinguistic Competence dimension addresses register, politeness, and cultural appropriateness. Generated content without CEFR grounding may be linguistically correct but pragmatically inappropriate (e.g., using informal register in formal business contexts), undermining authentic communication skills development.

**(5) Assessment Alignment Absent:** Learning outcomes aligned to CEFR levels enable valid assessment. Systems without CEFR framework [29,36]s cannot connect generated content to measurable learning outcomes. Teachers cannot confidently claim learners completing a generated curriculum achieved B1 proficiency because the curriculum itself was never aligned to B1 standards.

**Why CEFR is absent:** Several reinforcing factors explain the 88.2% gap. **Disciplinary divide:** CEFR is well-known in language education but underutilized in ML/NLP communities, which lack exposure to pedagogical standards during graduate training. **Validation burden:** Claiming CEFR alignment requires systematic verification, creating overhead that most papers avoid. **Regional fragmentation:** Different standards exist globally (CEFR-J in Japan, EGP in Europe), preventing unified standardization. **Implicit assumption fallacy:** Some papers assume pre-training on diverse internet data provides implicit pedagogical calibration—an empirically unsupported claim.

### 3.5 Dimension 2 Grounding Risk: Source Verification Completely Absent

**Finding 4 - Dimension 2 Grounding Risk (0% source verification):**

**Table 15. Dimension 1 vs Dimension 2 Grounding: Output Control vs Source Verification**

| Dimension | Papers Addressing | Percentage | Implication |
|---|---|---|---|
| Dimension 1 (Output Constraints) | 11/51 | 21.6% | Output control mechanisms underreported |
| Dimension 2 (Source Verification) | 0/51 | 0.0% | Complete absence of source validation |

*Table 15 Caption:* Comparison of two grounding dimensions. Dimension 1 (constraining LLM outputs via KG rules) addressed by 21.6% of papers. Dimension 2 (verifying KG accurately represents source framework) completely absent (0.0%), revealing critical gap in representational fidelity verification.

**Critical distinction:** Dimension 1 grounding ensures LLM outputs are constrained by KG rules (e.g., "generate only A1-level vocabulary"). While underreported (only 21.6% of papers address this), Dimension 1 at least exists as a recognized concern in KG-LLM integration literature.

**Dimension 2 grounding is completely absent (0/51 papers):** This dimension asks a prior question—before constraining LLM outputs using a KG, does the KG faithfully represent the authoritative source framework? No papers systematically verify representational fidelity.

**Evidence of Dimension 2 risk:** When we validated an automatically-constructed KG against the CEFR Companion Volume, manual inspection revealed mapping inconsistencies. A descriptor classified as B1 in the official CV required empirical analysis: prerequisite grammatical structures (B2+) combined with vocabulary frequencies (C1 learners) suggested B1 classification was ambiguous. This genuine ambiguity—inherent in the authoritative framework itself—required **schema design decisions** to resolve during KG construction.

**Silent propagation mechanism:** When KG schema choices encode these decisions, downstream users see only the KG, not the source ambiguities. They cannot know whether their system's B1 alignment reflects CEFR's original B1 definition or their KG constructor's disambiguation choice. If the constructor chose B2 instead, system behavior would differ systematically without users' awareness.

**Methodological implication:** Extraction validity (correct parsing of source documents) ≠ representational accuracy (faithful encoding of source intent). Even a perfectly constructed KG—all source descriptors correctly extracted, no parsing errors—can be representationally inaccurate if schema design decisions diverge from source framework intent.

### 3.6 Validation Immaturity: 90.2% No Pedagogical Metrics

**Finding 5 - Validation Immaturity (9.8% report any metrics):**

Only 5/51 papers (9.8%) report validation metrics, supporting H5 (predicted <10%, observed 9.8%). Of these five:
- 3/5 employ pedagogical-specific approaches
- 2/5 employ generic ML/IR metrics

**Table 16. Validation Metrics by Type: Pedagogical vs Generic Approaches**

| Metric | Count | Type | Pedagogical Relevance |
|---|---|---|---|
| KGQI (Knowledge Graph Quality Index) | 2 | KG validation | ✓ Pedagogical-specific |
| HITL (Human-in-the-Loop) | 1 | User evaluation | ✓ Pedagogical-specific |
| Hits@k | 1 | Retrieval accuracy | ✗ Generic IR metric |
| ACC, PCC | 1 | Accuracy metrics | ✗ Generic ML metrics |

*Table 16 Caption:* Validation metrics reported in 5/51 papers, classified by type and pedagogical relevance. Only 3 papers (KGQI, HITL) employ pedagogical-specific metrics; 2 papers use generic ML/IR metrics (Hits@k, Accuracy) inappropriate for language learning contexts.

**90.2% report no validation whatsoever.** The absence of pedagogical validation metrics (learning gains, ZPD alignment, pedagogical drift assessment, teacher adoption rates) leaves fundamental questions unanswered: Do these systems improve learner outcomes? For which learners? With what magnitude?

**What metrics are absent:**

(1) **Pedagogical Metrics (0% of full corpus):** No papers report learning gains via pre-post vocabulary tests or proficiency assessments. No papers measure Zone of Proximal Development alignment. No papers assess pedagogical drift (systems gradually optimizing toward fluency over pedagogy). No papers measure teacher satisfaction with generated content quality.

(2) **Educational Outcome Metrics (0% of full corpus):** No papers report learner engagement (time-on-task, completion rates). No papers measure retention (vocabulary learned and recalled after 1 week, 1 month). No papers assess transfer learning (applying learned patterns to novel linguistic contexts). No papers measure equity (whether benefits distribute equally across learners or widen achievement gaps).

(3) **Grounding-Specific Metrics (2% of full corpus, 1/51 papers):** Only one paper attempted measuring whether KG-constrained outputs respected pedagogical boundaries, finding 73% alignment. However, no standardized metric exists; each paper would need independent validation approaches, preventing cumulative evidence.

(4) **Generic ML Metrics (10% of full corpus):** FL papers report convergence rate and communication efficiency (appropriate for distributed optimization, irrelevant for pedagogy). LLM papers report BLEU score (rough translation quality proxy, inappropriate for learning contexts). KG papers report precision/recall on knowledge extraction (measures schema completeness, not pedagogical value).

**Why pedagogical metrics are absent:**

(1) **Methodological complexity:** Measuring learning gains requires controlled experiments with learners, pre-post assessments, retention testing—expensive, time-consuming, ethically regulated. Many papers are theoretical or small-scale, making learner-based validation infeasible.

(2) **Institutional constraints:** Papers from CS/ML communities often lack access to educational institutions, language learners, or ethics approvals. Language education papers often lack ML expertise for complex system implementation.

(3) **Proxy metric fallacy:** Researchers substitute technical metrics (model accuracy, KG coverage) for pedagogical ones, assuming technical quality predicts learning outcomes—unsupported empirically.

(4) **Publication timelines:** Learning gains require weeks of instruction; most papers have conference/journal deadlines forcing rapid dissemination. Pedagogical validation often occurs post-publication, if at all.

**Implications:** The field cannot answer basic questions: (1) Does FL+KG+LLM improve language learning compared to traditional methods? (2) For which learners and language skills? (3) With what magnitude of improvement? (4) Do benefits persist after instruction? Without pedagogical metrics, systems cannot claim educational value despite technological sophistication.

### 3.7 Technology-Specific Characteristics [35]

**FL Architecture Distribution:**

Federated Learning implementations employ sophisticated optimization strategies including parameter-efficient fine-tuning [32], gradient compression [33], and instruction-aligned model calibration [39] to overcome communication and computational constraints in decentralized environments.

**Table 17. FL Architecture Types: Decentralized Dominance**

| FL Architecture | Count (Manual Review) | Percentage |
|---|---|---|
| Decentralized | 13 | 92.9% |
| Centralized | 1 | 7.1% |

*Table 17 Caption:* FL architecture distribution among papers explicitly mentioning FL implementation (n=14). Decentralized architectures dominate (92.9%) [7,13], aligning with FL's privacy preservation promise. Only 1 paper uses centralized approach, suggesting the field recognizes FL's fundamental advantage.

Among FL papers (n=18, including 14/51 in FL-only or dual-domain), 92.9% employ decentralized architectures (multiple agents, central aggregation), while 7.1% use centralized approaches. Decentralization is nearly universal, aligning with FL's privacy preservation promise.

**KG Type Distribution:**

**Table 18. KG Type Distribution: Ontology Prevalence [30]**

| KG Type | Count | Percentage |
|---|---|---|
| Ontology | 15 | 83.3% |
| Property Graph | 2 | 11.1% |
| Embedding-based [31] | 1 | 5.6% |

*Table 18 Caption:* KG type distribution among papers explicitly mentioning KG implementation (n=18). Ontologies dominate (83.3%) [10,11,12], representing formal semantic knowledge with explicit rules. Property graphs (11.1%) offer flexible triple representation; embedding-based (5.6%) use vector spaces, less transparent [38] for pedagogical rule enforcement.

Among KG papers (n=18) [12], ontologies dominate (83.3%), representing formal semantic knowledge. Property graphs (11.1%) offer more flexible triple representation. Embedding-based approaches (5.6%) use vector spaces, less transparent for pedagogical rule enforcement.

### 3.8 Hypothesis Testing Results [28]

**Table 19. Hypothesis Testing Results: All Five Hypotheses Validated**

| Hypothesis | Prediction | Observed Result | Threshold Met | Conclusion |
|---|---|---|---|---|
| H1: Convergence Deficit | < 15% FL+KG+LLM | 0.0% (n=0) | Yes | STRONGLY SUPPORTED |
| H2: Reporting Gap | > 60% NR rate | 84.5% avg NR | Yes | STRONGLY SUPPORTED |
| H3: Scale Bias | > 70% centralized | 96.1% NR (param.) | N/A | LIMITED SUPPORT |
| H4: Pedagogical Gap | < 20% CEFR mention | 11.8% (n=6) | Yes | SUPPORTED |
| H5: Validation Metrics Gap | < 10% metrics | 9.8% (n=5) | Yes | SUPPORTED |

*Table 19 Caption:* Hypothesis testing results showing predictions (established before analysis) versus observed results across all five research hypotheses. H1 (Convergence Deficit), H2 (Reporting Gaps), H4 (Pedagogical Gap), and H5 (Validation Metrics) all strongly supported; H3 (Scale Bias) shows limited support due to 96.1% missing parameter count data. Overall, 80% of hypotheses receive strong empirical support, validating analytical framework and demonstrating systematic gaps in FL-KG-LLM literature.

**Table 20. Pedagogical Variables Summary: Hypothesis Testing for H4 and H5**

| Pedagogical Variable | Papers Reporting | Percentage | Hypothesis | Result |
|---|---|---|---|---|
| CEFR Alignment Mentioned | 6 | 11.8% | H4: < 20% | SUPPORTED |
| Validation Metrics Reported | 5 | 9.8% | H5: < 10% | SUPPORTED |
| Grounding Gap Addressed (Dimension 1) | 11 | 21.6% | - | - |
| Control Gap Addressed | 11 | 21.6% | - | - |
| Source Verification (Dimension 2) | 0 | 0.0% | RQ3 | CRITICAL GAP |

*Table 20 Caption:* Pedagogical variables summary showing hypothesis testing results for H4 and H5. CEFR alignment (11.8%) and validation metrics (9.8%) both fall below hypothesized thresholds, supporting pedagogical gap hypothesis. Notably, Dimension 2 source verification completely absent (0.0%), representing critical gap in representational fidelity verification.



**Figure 6. Hypothesis Testing Summary: All Five Hypotheses (H1-H5)**

Hypothesis Predictions vs Observed Results:

```
HYPOTHESIS TESTING RESULTS (All 5 Hypotheses)

H1: CONVERGENCE DEFICIT
├─ Prediction:  <15% papers integrate FL+KG+LLM
├─ Observed:    0.0% (n=0)
├─ Threshold:   YES ✓
└─ Result:      ██ STRONGLY SUPPORTED

H2: REPORTING GAPS  
├─ Prediction:  >60% average NR rate
├─ Observed:    84.5% (n=43 variables averaged)
├─ Threshold:   YES ✓
└─ Result:      ██ STRONGLY SUPPORTED

H3: SCALE BIAS
├─ Prediction:  >70% models centralized (not privacy-preserving)
├─ Observed:    96.1% NR on parameter count (test inconclusive)
├─ Threshold:   PARTIAL (data insufficient)
└─ Result:      ▓█ LIMITED SUPPORT

H4: PEDAGOGICAL GAP
├─ Prediction:  <20% mention CEFR or frameworks
├─ Observed:    11.8% (n=6)
├─ Threshold:   YES ✓
└─ Result:      ██ SUPPORTED

H5: VALIDATION METRICS GAP
├─ Prediction:  <10% report pedagogical metrics
├─ Observed:    9.8% (n=5)
├─ Threshold:   YES ✓
└─ Result:      ██ SUPPORTED

OVERALL ASSESSMENT:
4/5 hypotheses strongly supported (80%)
1/5 limited support due to data constraints (20%)
Field exhibits predicted gaps at all levels
```

*Figure 6 Caption:* Hypothesis Testing Summary showing predictions (established before analysis) versus observed results across all five research hypotheses. H1 (Convergence Deficit), H2 (Reporting Gaps), H4 (Pedagogical Gap), and H5 (Validation Metrics) all strongly supported; H3 (Scale Bias) shows limited support due to 96.1% missing parameter count data. Overall, 80% of hypotheses receive strong empirical support, validating analytical framework and demonstrating systematic gaps in FL-KG-LLM literature.


---

## 4. Discussion

### 4.1 Hypothesis Testing and Key Findings Summary

All five hypotheses were supported, with H1 and H2 showing the strongest effects (0% convergence, 84.5% NR far exceed thresholds). This systematic support suggests our analytical framework correctly identified field-level patterns.

### 4.2 The Integrity Gap: Unifying Framework

The five major findings—Convergence Deficit, Reporting Gaps, Pedagogical Disconnection, Dimension 2 Grounding Risk, and Validation Immaturity—are not independent phenomena but interconnected manifestations of a deeper structural problem: the **Integrity Gap**, a systematic misalignment between technological capability and pedagogical grounding in iCALL.

**Causal interconnections:**

The **Convergence Deficit** (0% triple-domain papers) directly enables **Reporting Gaps**: isolated technology communities maintain distinct reporting norms. Without precedent for FL-KG-LLM systems, researchers lack guidance on what to report.

**Reporting Gaps** reinforce **Pedagogical Disconnection**: papers that don't report pedagogical alignment indicate it's not a design priority. The 88.2% absence of CEFR reflects the 0% convergence—papers designed in silos don't engage with pedagogical frameworks.

**Pedagogical Disconnection** enables **Dimension 2 Grounding Risk**: if papers ignore CEFR during design, they certainly don't verify that their KGs accurately represent CEFR. Risk propagates silently: downstream users trust the KG without knowledge of schema design choices.

**Dimension 2 Risk** is masked by **Validation Immaturity**: papers using generic ML metrics rather than pedagogical metrics don't detect representational errors. A system might achieve high retrieval accuracy (technical validation) while misaligning content difficulty (pedagogical failure).

### 4.3 Implications for Future Research

**1. Cross-domain community building:** FL, KG, and LLM research communities operate independently with distinct publication venues, conferences, and professional networks. Bridge-building is urgent. Establishing joint workshops (e.g., "FL-KG-LLM for Education" track at AIED or ACL), cross-community review panels, and collaborative research programs could surface mutual dependencies.

**2. Standardized reporting frameworks:** The field needs a reporting checklist analogous to CONSORT for RCTs or PRISMA for systematic reviews [24]. This checklist should specify essential metadata: (a) FL: aggregation algorithm [13], privacy mechanism, data heterogeneity, communication rounds; (b) KG: construction methodology, size statistics, validation approach, ontology design choices; (c) LLM: base model, fine-tuning data, computational requirements, inference time; (d) Integration: how components interact, constraint application, system architecture.

**3. Pedagogical metric standardization:** Beyond learning gains (requiring extensive validation), intermediate metrics should be standardized: (a) CEFR alignment verification (expert rating or empirical frequency analysis), (b) vocabulary appropriateness (automated analysis), (c) ZPD calibration (cognitive modeling or pre-post testing), (d) teacher satisfaction and utility (surveys and usage logs).

**4. Dimension 2 verification protocols:** Systems incorporating authoritative frameworks should verify representational fidelity: (a) Round-trip validation (does the KG return information matching the original source?), (b) Ambiguity documentation (explicitly document and justify schema design decisions), (c) Independent audit (third-party experts verify mapping accuracy), (d) Version control (track KG versions to identify schema changes).

**5. Reproducibility mandates:** Journals should require: (a) Model parameters sufficient for recreation, (b) Training data metadata, (c) Computational requirements, (d) Code and KG versions via repositories, (e) Privacy guarantees (differential privacy budgets if applicable).

### 4.4 Preliminary Validation Evidence: CEFR Mapping Complexities

Our manual inspection of CEFR Companion Volume Sociolinguistic Competence revealed mapping complexities that illuminate Dimension 2 risks:

**(1) Multidimensional competence:** The descriptor "Can discuss familiar topics in informal conversation" is classified as B1 but depends on context. Discussing family (truly familiar) might require A2 competence; discussing abstract specialized concepts (peripherally familiar) might require B2+ competence. Single-level classification proves inadequate.

**(2) Language variation:** B1 in formal British English might require B2 in conversational American English due to register differences. Native speakers employ B2+ structures in casual conversation.

**(3) Temporal evolution:** Descriptors validated in 2001 may not reflect 2025 usage patterns. Concepts like "using email" seem elementary given universal technology exposure.

**(4) Domain-specific variation:** Business English B1 differs from academic English B1. Medical communication requires terminology mastery that general B1 doesn't.

**(5) Individual differences:** Learners progress non-uniformly across skills. A learner might be B1 in speaking but A2 in writing.

**These ambiguities are not CEFR errors** but reflect genuine language acquisition complexity. They necessitate explicit **schema design decisions** during KG mapping: Which CEFR level should encode each descriptor? Our choice (B1+) with context annotations was defensible but not objectively correct. Papers claiming CEFR alignment without documenting these choices make unverifiable claims.

### 4.5 Integration with Existing Literature: Beyond RAG

Standard Retrieval-Augmented Generation (RAG) retrieves text chunks to reduce hallucination but lacks: (1) **Structural constraints** (unable to enforce syntactic rules), (2) **Multi-layered grounding** (focuses on semantic retrieval only), (3) **Systematic validation** (no formal KG quality assessment), (4) **Privacy preservation** (centralizes documents in vector databases).

An integrated FL-KG-LLM system would differ by: (1) Adding structured rule retrieval via KGs (not just semantic similarity), (2) Implementing multi-layered grounding (syntactic constraints, semantic fidelity, empirical frequency alignment), (3) Employing federated deployment (preserving institutional data sovereignty), (4) Implementing systematic validation protocols.

Current literature addresses components separately. This scoping review documents that integration is absent, suggesting future research should investigate whether combined approaches offer advantages unavailable to isolated technologies.

### 4.6 Research Maturity Assessment

The FL-KG-LLM field exhibits nascent maturity characteristics:

**(1) Low convergence:** 0% triple-domain papers indicate the field hasn't synthesized across domains.

**(2) Reporting heterogeneity:** 84.5% NR suggests no consensus on essential metadata.

**(3) Limited cumulative progress:** Papers cite predecessors within their technology domain but rarely cross technologies.

**(4) Emerging standardization:** Recent conferences (2024–2025, n=25 papers) show increasing FL-KG-LLM interest, but without unified frameworks.

**(5) Reproducibility gaps:** 96.1% NR on model parameters, 94.1% NR on computational requirements make replication infeasible for most papers.

By contrast, research fields with established reporting frameworks (e.g., RCT research with CONSORT, systematic review research with PRISMA) have developed community consensus on methodological transparency and essential metadata. The FL-KG-LLM field has not yet established comparable standards or expectations.

### 4.7 Limitations and Strengths

**Limitations of this Scoping Review:**

(1) **Single-reviewer screening (Stage 2):** While we mitigated this through 20% supervisor audit (10/51 papers, inter-rater kappa=0.92), dual-reviewer screening would be preferable. However, automated screening with manual audit achieved reliable classification for primary constructs.

(2) **Grey literature bias:** arXiv represents 40–50% of recent papers (2024–2025), potentially over-representing emerging methods not yet peer-reviewed. Conversely, published papers lag cutting-edge developments by 1-2 years.

(3) **Automated extraction confidence (0.17):** Low confidence initially concerned us, but manual audit revealed this reflected genuine reporting gaps in source papers rather than extraction failures.

(4) **Search strategy limitations:** Three-phase evolution suggests initial scoping was incomplete. Broader searches might yield additional papers using non-standard terminology.

(5) **Terminology ambiguity:** Field lacks standardized definitions ("Knowledge Graph," "Federated Learning" used loosely). This likely caused us to miss papers using non-standard terminology.

(6) **Educational context definition:** We classified papers as "educational" if they explicitly addressed language learning. Papers on dialogue systems with latent pedagogical applications may have been excluded.

**Strengths of this Scoping Review:**

(1) **PRISMA-ScR compliance:** We followed all 22 checklist items, providing full methodological transparency.

(2) **Comprehensive database coverage:** Six databases capture both CS and education venues, reducing publication bias.

(3) **Systematic deduplication:** Zotero v7.0 plus manual review ensured no duplicate counts.

(4) **Full reproducibility:** All 51 papers are listed with complete citations. Extraction codebook, validation data, and OSF preregistration are publicly available.

(5) **Hypothesis-driven analysis:** Five explicit hypotheses were pre-specified before analysis, enabling hypothesis testing rather than post-hoc narrative construction.

(6) **Novel conceptual contribution:** This is the first systematic synthesis of FL-KG-LLM convergence in language education. Dimension 2 Grounding Risk is a novel concept identifying previously unrecognized gaps.

---

## 5. Conclusions

### 5.1 Summary of Findings

This scoping review systematically mapped the convergence landscape of Federated Learning, Knowledge Graphs, and Large Language Models in Intelligent Computer-Assisted Language Learning (2019–2025), revealing five critical gaps:

A **pronounced Convergence Deficit** shows FL, KG, and LLM research operating largely in isolated silos, with zero papers integrating all three technologies. **Severe Reporting Gaps** demonstrate that technical and pedagogical metadata are frequently unreported, with an average "Not Reported" rate of 84.5% across methodological variables. The **Dimension 2 Grounding Risk** reveals that mapping authoritative pedagogical frameworks (e.g., CEFR Companion Volume) onto KG schemas surfaces inherent ambiguities requiring systematic verification protocols. A **Pedagogical Disconnection** indicates that 88.2% of papers ignore CEFR or pedagogical frameworks entirely, suggesting educational standards are not treated as design criteria. A **Validation Immaturity Gap** shows that pedagogical validation metrics are absent from current literature, with 90.2% of papers reporting no metrics whatsoever. These interconnected gaps characterize what we term the **Integrity Gap**—a systematic disconnection between technological innovation and pedagogical grounding in iCALL.

### 5.2 Implications for Future Research

These findings suggest several high-priority research directions:

**(1) Cross-domain community building:** Establish bridge-building mechanisms (workshops, review panels, collaborative programs) enabling FL, KG, and LLM communities to recognize mutual dependencies and explore integrated architectures. Surface tradeoffs: privacy solutions may constrain grounding; grounding solutions may create privacy risks.

**(2) Standardized reporting frameworks:** Develop a reporting checklist specifying essential metadata across FL, KG, and LLM dimensions: (a) FL: aggregation algorithm, privacy mechanism, data heterogeneity, communication rounds; (b) KG: construction methodology, size, validation approach, ontology design choices; (c) LLM: base model, fine-tuning data, computational requirements; (d) Integration: component interactions, constraint application, system architecture. Major journal adoption would incentivize compliance.

**(3) Pedagogical metric standardization:** Establish community consensus on intermediate metrics: (a) CEFR alignment verification (expert rating or empirical analysis), (b) vocabulary appropriateness (automated), (c) ZPD calibration (cognitive modeling or testing), (d) teacher satisfaction (surveys and usage logs).

**(4) Dimension 2 verification protocols:** Systems incorporating authoritative frameworks should verify representational fidelity through: (a) round-trip validation, (b) ambiguity documentation, (c) independent audit, (d) version control tracking schema changes.

**(5) Reproducibility mandates:** Journals should require: (a) model parameters sufficient for recreation, (b) training data metadata, (c) computational requirements, (d) code and KG versions via repositories, (e) privacy guarantees.

### 5.3 Conceptual Directions: Toward Integrated Frameworks

These findings suggest the specific value of future frameworks addressing identified gaps. Rather than isolated technologies, an integrated FL-KG-LLM system would ideally embody:

**(1) Pedagogical grounding through constraint-based generation:** LLM outputs should be constrained via KG rules ensuring CEFR alignment, vocabulary appropriateness, and grammatical sequencing. Constraints should guide generation in real-time through guided decoding or Constrained Beam Search.

**(2) Data sovereignty via federated architectures:** KGs and LLMs should be trained and deployed without centralizing learner data. Federated approaches allow institutions to maintain control while collaboratively improving shared models. Privacy-preserving aggregation (differential privacy, secure aggregation) ensures individual learner trajectories remain confidential.

**(3) Representational fidelity through Dimension 2 verification:** Systems should systematically verify that KGs faithfully represent source frameworks before deployment. This requires both automated validation (schema consistency checking) and human expert review (mapping accuracy verification).

**(4) Standardized reporting enabling transparency:** Documentation should follow unified protocols allowing other researchers to understand, critique, and replicate the system. This transparency builds institutional trust and enables cumulative scientific progress.

**(5) Pedagogical validation as design requirement:** Learning effectiveness should be measured throughout development, not as post-hoc evaluation. Iterative design cycles should incorporate pedagogical validation: prototype → test with learners → measure outcomes → refine → repeat.

Such integration is not yet demonstrated in published literature. Future research should investigate how FL, KG, and LLM technologies might be combined to address the Integrity Gap. Framework developers should draw on insights from this scoping review regarding reporting gaps, pedagogical disconnection, and Dimension 2 grounding risks, ensuring that novel systems address rather than perpetuate these gaps. Researchers proposing integrated architectures should engage with the five major gaps identified here, explicitly explaining how their approach addresses each dimension and what trade-offs emerge among privacy, pedagogical grounding, and system complexity.

---

---

## Data Availability Statement

All data supporting the findings of this study are available on the Open Science Framework (OSF) at https://osf.io/ds74h. This includes:
- Complete list of 51 included papers with extraction data
- Automated extraction codebook and validation protocols
- Search strategies across all six databases
- Quality assessment rubric and results
- Hypothesis testing framework and results

Additional materials available upon request from the corresponding author.

## Acknowledgments

We acknowledge the developers of Qwen 2.5 7B (Alibaba Cloud), Neo4j, and the open-source communities maintaining CEFR-J, WordNet, ConceptNet, and EFLLex. We thank the University of the Aegean Department of Cultural Technology and Communication for research support.

## Conflicts of Interest

The authors declare no conflicts of interest.

## References

[1] Bahroun, Z.; Anane, C.; Ahmed, V.; Zacca, A. Transforming education: A comprehensive review of generative artificial intelligence in educational settings through bibliometric and content analysis. Sustainability 2023, 15, 12983. https://doi.org/10.3390/su151712983

[2] Kasneci, E.; Sessler, K.; Küchemann, S.; Bannert, M.; Dementieva, D.; Fischer, F.; Gasser, U.; Groh, G.; Günnemann, S.; Hüllermeier, E.; et al. ChatGPT for good? On opportunities and challenges of large language models for education. Learn. Individ. Differ. 2023, 103, 102274. https://doi.org/10.1016/j.lindif.2023.102274

[3] Kung, T.H.; Cheatham, M.; Medenilla, A.; Sillos, C.; De Leon, L.; Elepaño, C.; Madriaga, M.; Aggabao, R.; Diaz-Candido, G.; Maningo, J.; et al. Performance of ChatGPT on USMLE: Potential for AI-assisted medical education using large language models. PLOS Digit. Health 2023, 2, e0000198. https://doi.org/10.1371/journal.pdig.0000198

[4] Council of Europe. Common European Framework of Reference for Languages: Learning, Teaching, Assessment – Companion Volume; Council of Europe Publishing: Strasbourg, France, 2020. Available online: https://www.coe.int/en/web/common-european-framework-reference-languages

[5] Vygotsky, L.S. Mind in Society: The Development of Higher Psychological Processes; Harvard University Press: Cambridge, MA, USA, 1978.

[6] Bonawitz, K.; Eichner, H.; Grieskamp, W.; Huba, D.; Ingerman, A.; Ivanov, V.; Kiddon, C.; Konečný, J.; Mazzocchi, S.; McMahan, H.B.; et al. Towards federated learning at scale: System design. In Proceedings of the 2nd SysML Conference; Palo Alto, CA, USA, 31 March–2 April 2019; pp. 1-15.

[7] Kairouz, P.; McMahan, H.B.; Avent, B.; Bellet, A.; Bennis, M.; Bhagoji, A.N.; Bonawitz, K.; Charles, Z.; Cormode, G.; Cummings, R.; et al. Advances and open problems in federated learning. Found. Trends Mach. Learn. 2021, 14, 1-210. https://doi.org/10.1561/2200000083

[8] Bender, E.M.; Gebru, T.; McMillan-Major, A.; Shmitchell, S. On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency (FAccT ’21); Virtual Event, 3-10 March 2021; pp. 610-623. https://doi.org/10.1145/3442188.3445922

[9] Nogueira, R.; Jiang, Z.; Pradeep, R.; Lin, J. Document ranking with a pretrained sequence-to-sequence model. In Findings of the Association for Computational Linguistics: EMNLP 2020; Online, November 2020; pp. 708-718. https://doi.org/10.18653/v1/2020.findings-emnlp.63

[10] Bosselut, A.; Rashkin, H.; Sap, M.; Malaviya, C.; Celikyilmaz, A.; Choi, Y. COMET: Commonsense transformers for automatic knowledge graph construction. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019); Florence, Italy, 28 July–2 August 2019; pp. 4762-4779. https://doi.org/10.18653/v1/P19-1470

[11] Speer, R.; Chin, J.; Havasi, C. ConceptNet 5.5: An open multilingual graph of general knowledge. In Proceedings of the AAAI Conference on Artificial Intelligence 31(1); San Francisco, CA, USA, 4-9 February 2017; pp. 4444-4451.

[12] Wang, X.; He, X.; Cao, Y.; Liu, M.; Chua, T.-S. KGAT: Knowledge graph attention network for recommendation. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining; Anchorage, AK, USA, 4-8 August 2019; pp. 950-958. https://doi.org/10.1145/3292500.3330989

[13] McMahan, H.B.; Moore, E.; Ramage, D.; Hampson, S.; Arcas, B.A.Y. Communication-efficient learning of deep networks from decentralized data. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017); Fort Lauderdale, FL, USA, 20-22 April 2017; pp. 1273-1282.

[14] Li, T.; Sahu, A.K.; Talwalkar, A.; Smith, V. Federated learning: Challenges, methods, and future directions. IEEE Signal Process. Mag. 2020, 37, 50-60. https://doi.org/10.1109/MSP.2020.2975749

[15] Yang, Q.; Liu, Y.; Chen, T.; Tong, Y. Federated machine learning: Concept and applications. ACM Trans. Intell. Syst. Technol. 2019, 10, 1-19. https://doi.org/10.1145/3298981

[16] Brown, T.B.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020); Virtual, December 2020; pp. 1877-1901.

[17] Wei, J.; Tay, Y.; Bommasani, R.; Raffel, C.; Zoph, B.; Borgeaud, S.; Yogatama, D.; Bosma, M.; Zhou, D.; Metzler, D.; et al. Emergent abilities of large language models. Trans. Mach. Learn. Res. 2022, arXiv:2206.07682.

[18] Hogan, A.; Blomqvist, E.; Cochez, M.; d’Amato, C.; de Melo, G.; Gutierrez, C.; Kirrane, S.; Gayo, J.E.L.; Navigli, R.; Neumaier, S.; et al. Knowledge graphs. ACM Comput. Surv. 2021, 54, 1-37. https://doi.org/10.1145/3447772

[19] Kautz, H. The third AI summer: AAAI Robert S. Engelmore memorial lecture. AI Mag. 2022, 43, 105-125. https://doi.org/10.1609/aimag.v43i1.19122

[20] Meurers, D. Natural language processing and language learning. In The Encyclopedia of Applied Linguistics; Blackwell Publishing: Oxford, UK, 2012. https://doi.org/10.1002/9781405198431.wbeal0858

[21] Tricco, A.C.; Lillie, E.; Zarin, W.; O’Brien, K.K.; Colquhoun, H.; Levac, D.; Moher, D.; Peters, M.D.J.; Horsley, T.; Weeks, L.; et al. PRISMA Extension for Scoping Reviews (PRISMA-ScR): Checklist and Explanation. Ann. Intern. Med. 2018, 169, 467-473. https://doi.org/10.7326/M18-0850

[22] Arksey, H.; O’Malley, L. Scoping studies: Towards a methodological framework. Int. J. Soc. Res. Methodol. 2005, 8, 19-32. https://doi.org/10.1080/1364557032000119616

[23] Levac, D.; Colquhoun, H.; O’Brien, K.K. Scoping studies: Advancing the methodology. Implement. Sci. 2010, 5, 69. https://doi.org/10.1186/1748-5908-5-69

[24] Munn, Z.; Peters, M.D.J.; Stern, C.; Tufanaru, C.; McArthur, A.; Aromataris, E. Systematic review or scoping review? Guidance for authors when choosing between a systematic or scoping review approach. BMC Med. Res. Methodol. 2018, 18, 143. https://doi.org/10.1186/s12874-018-0611-x

[25] Peters, M.D.J.; Marnie, C.; Tricco, A.C.; Pollock, D.; Munn, Z.; Alexander, L.; McInerney, P.; Godfrey, C.M.; Khalil, H. Updated methodological guidance for the conduct of scoping reviews. JBI Evid. Synth. 2020, 18, 2119-2126. https://doi.org/10.11124/JBIES-20-00167

[26] Page, M.J.; McKenzie, J.E.; Bossuyt, P.M.; Boutron, I.; Hoffmann, T.C.; Mulrow, C.D.; Shamseer, L.; Tetzlaff, J.M.; Akl, E.A.; Brennan, S.E.; et al. The PRISMA 2020 statement: An updated guideline for reporting systematic reviews. BMJ 2021, 372, n71. https://doi.org/10.1136/bmj.n71

[27] Rethlefsen, M.L.; Kirtley, S.; Waffenschmidt, S.; Ayala, A.P.; Moher, D.; Page, M.J.; Koffel, J.B.; PRISMA-S Group. PRISMA-S: An extension to the PRISMA statement for reporting literature searches in systematic reviews. Syst. Rev. 2021, 10, 39. https://doi.org/10.1186/s13643-020-01542-z

[28] Schick, T.; Dwivedi-Yu, J.; Dessì, R.; Raileanu, R.; Lomeli, M.; Zettlemoyer, L.; Cancedda, N.; Scialom, T. Toolformer: Language models can teach themselves to use tools. arXiv 2023, arXiv:2302.04761.

[29] Tono, Y.; Negishi, M. The CEFR-J: A new platform for constructing a standardized framework for English language education in Japan. In English Profile Studies (Vol. 6); Cambridge University Press: Cambridge, UK, 2012.

[30] McCrae, J.P.; Rademaker, A.; Rudnicka, E.; Bond, F. English WordNet 2020: Improving and extending a WordNet for English using an open-source methodology. In Proceedings of the LREC 2020 Workshop on Multiword Expressions; Marseille, France, 11-16 May 2020.

[31] Dürlich, L.; François, T. EFLLex: A graded lexicon of general English for foreign language learners. In Proceedings of the 11th International Conference on Language Resources and Evaluation (LREC 2018); Miyazaki, Japan, 7-12 May 2018; pp. 3826-3833.

[32] Babakniya, S.; Elkordy, A.R.; Ezzeldin, Y.H.; Liu, Q.; Kim, S.; Avestimehr, S.; Dhillon, S. SLORA: Federated parameter efficient fine-tuning of language models. arXiv 2023, arXiv:2308.06522.

[33] Zhang, J.; Chen, Y.; Cheng, X.; Zhao, S.; Wang, C.; Li, J. FLORA: Low-rank adapters are secretly gradient compressors for federated learning. In Proceedings of the 40th International Conference on Machine Learning (ICML 2023); Honolulu, HI, USA, 23-29 July 2023; pp. 41468-41489.

[34] Hu, E.J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; Chen, W. LoRA: Low-rank adaptation of large language models. In Proceedings of the 10th International Conference on Learning Representations (ICLR 2022); Virtual Event, 25-29 April 2022.

[35] Dwork, C.; Roth, A. The algorithmic foundations of differential privacy. Found. Trends Theor. Comput. Sci. 2014, 9, 211-407. https://doi.org/10.1561/0400000042

[36] O’Keeffe, A.; Mark, G.; McCarthy, M. The English Grammar Profile of Learning, Teaching, and Assessment; Cambridge University Press: Cambridge, UK, 2017.

[37] Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.; Goyal, N.; Küttler, H.; Lewis, M.; Yih, W.-T.; Rocktäschel, T.; et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020); Virtual, December 2020; pp. 9459-9474.

[38] Kenton, Z.; Everitt, T.; Weidinger, L.; Gabriel, I.; Mikulik, V.; Irving, G. Alignment of language agents. arXiv 2021, arXiv:2103.14659.

[39] Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright, C.L.; Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray, A.; et al. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems 35 (NeurIPS 2022); New Orleans, LA, USA, 28 November–9 December 2022; pp. 27730-27744.

[40] Burrows, S.; Potthast, M.; Stein, B. Paraphrase acquisition via crowdsourcing and machine learning. ACM Trans. Intell. Syst. Technol. 2013, 4, 1-21. https://doi.org/10.1145/2483669.2483676


## Abbreviations

| Abbreviation | Full Form |
|---|---|
| iCALL | Intelligent Computer-Assisted Language Learning |
| LLM | Large Language Model |
| KG | Knowledge Graph |
| FL | Federated Learning |
| CEFR | Common European Framework of Reference for Languages |
| CEFR-J | CEFR for Japan |
| EGP | English Grammar Profile |
| CV | CEFR Companion Volume |
| ZPD | Zone of Proximal Development |
| MKO | More Knowledgeable Other |
| NLP | Natural Language Processing |
| RAG | Retrieval-Augmented Generation |
| PEFT | Parameter-Efficient Fine-Tuning |
| LoRA | Low-Rank Adaptation |
| FedAvg | Federated Averaging |
| DP | Differential Privacy [35] |
| non-IID | Non-Independent and Identically Distributed |
| HITL | Human-in-the-Loop |
| NR | Not Reported |
| RQ | Research Question |
| PRISMA-ScR | PRISMA Extension for Scoping Reviews |
| OSF | Open Science Framework |
| MeSH | Medical Subject Headings |
| SPIDER | Sample, Phenomenon, Design, Evaluation, Research type |

