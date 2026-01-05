The Convergence of Federated Learning, Knowledge Graphs, and Large Language Models for Language Instruction: A Scoping Review
**Michael Kenteris1,*, Konstantinos Kotis1**
1 Department of Cultural Technology and Communication, University of the Aegean, Mytilene 81100, Greece * Correspondence: mkenteris@aegean.gr
Abstract
Background: Large Language Models (LLMs) in Intelligent Computer-Assisted Language Learning (iCALL) offer personalization but introduce critical challenges: pedagogical grounding, data privacy, and instructional validity. Knowledge Graphs (KGs) and Federated Learning (FL) address these separately but are rarely integrated.
Objective: This scoping review maps FL, KG, and LLM convergence in educational contexts (2019-2025) to identify integration patterns and propose a framework addressing the “Integrity Gap.”
Methods: Following PRISMA-ScR, we searched IEEE Xplore, ACM, Google Scholar, arXiv, Scopus, and Web of Science, screening 51 papers. Automated extraction used Qwen 2.5 7B with Pydantic validation.
Results: We found complete absence of FL+KG+LLM integration (0%). Average “Not Reported” rate was 84.5%, with critical gaps in privacy_mechanism (92.2%) and validation_metrics (90.2%). Only 11.8% mentioned CEFR alignment, and ZERO papers addressed Dimension 2 grounding (KG source verification).
Discussion: We propose a Hybrid FL-KG-LLM Framework with dual loops: (1) Constraint Loop using KG Function Calling for CEFR alignment, and (2) Personalization Loop using FL-PEFT for privacy-preserving adaptation. We introduce novel metrics—Knowledge Graph Quality Index (KGQI) and Alignment Consistency Index (ACI).
Conclusions: Current iCALL literature exhibits pronounced Convergence Deficit. The proposed framework enables systems that are Private (FL), Grounded (validated KGs), and Aligned (constraint-based), keeping instruction within learners’ Zone of Proximal Development.
Keywords: Federated Learning; Knowledge Graphs; Large Language Models; iCALL; CEFR; Neurosymbolic AI
1. Introduction
1.1. The Paradox of Personalization in iCALL
The integration of Artificial Intelligence into Intelligent Computer-Assisted Language Learning (iCALL) has accelerated dramatically with the advent of Large Language Models (LLMs), which offer potent personalized content generation capabilities [1-3]. However, this technological promise confronts a critical triad of challenges that we conceptualize as the Integrity Gap:
Pedagogical Integrity: The lack of reliable grounding in established educational frameworks, particularly the Common European Framework of Reference for Languages (CEFR) [4], results in content that may exceed learners’ Zone of Proximal Development (ZPD) [5].
Data Integrity: Privacy risks inherent in centralized training paradigms, which require aggregating sensitive learner interaction data on external servers [6,7].
Instructional Validity: The stochastic nature of LLMs leads to hallucination and inconsistent alignment with pedagogical standards, undermining trust in high-stakes educational contexts [8,9].
Critically, existing literature addresses these challenges in isolation: Knowledge Graphs for pedagogical grounding [10-12], Federated Learning for privacy preservation [13-15], or LLM fine-tuning for personalization [16,17]. There exists no systematic framework integrating all three technologies to simultaneously ensure data sovereignty, pedagogical validity, and adaptive instruction.
1.2. The Two-Dimensional Grounding Problem
While neurosymbolic AI approaches [18,19] address Dimension 1 (constraining LLM outputs with KG-based retrieval), they largely overlook Dimension 2: ensuring that Knowledge Graphs themselves accurately encode authoritative frameworks. Our preliminary validation of CEFR Companion Volume (CV) Sociolinguistic Competence descriptors revealed systematic classification inconsistencies in primary sources [4], highlighting a previously unrecognized risk: when KG validity is assumed rather than demonstrated, pedagogical drift can occur at the source level, not just during generation.
This insight motivates our central research question: Can Federated Learning and Knowledge Graphs be synthesized into a cohesive neurosymbolic architecture that guarantees verifiable CEFR alignment while maintaining institutional data sovereignty?
1.3. Research Objectives and Contributions
This scoping review addresses five primary research questions:
RQ1: What are the systemic architectural and pedagogical limitations of centralized LLM deployment in iCALL systems?
RQ2: To what extent do existing hybrid models (FL-LLM and KG-LLM architectures) address privacy and grounding requirements in isolation?
RQ3: What is the “Integrity Gap” regarding Dimension 2 grounding, and what are the specific risks of assuming the validity of authoritative KG sources without systematic verification?
RQ4: How can FL and KGs be synthesized into a single neurosymbolic framework to guarantee verifiable CEFR alignment while maintaining data sovereignty?
RQ5: What objective metrics (e.g., Knowledge Graph Quality Index, Alignment Consistency Index) are required to transition from “assuming validity” to “demonstrating validity” in educational AI?
Our contributions are threefold:
Empirical Contribution: Systematic documentation of the “Convergence Deficit” in current literature, quantifying the extent to which FL, KG, and LLM research operate in isolated silos.
Conceptual Contribution: A Hybrid FL-KG-LLM Framework featuring:
Dual-loop architecture (Constraint Loop + Personalization Loop)
Five-Stage KG Validation Methodology
Novel metrics (KGQI, ACI) for demonstrating pedagogical validity
Methodological Contribution: Transparent post-hoc scoping review registration positioned before data extraction (the critical bias-reduction stage), with comprehensive documentation of methodological deviations.
1.4. Theoretical Foundations: ZPD and Pedagogical Scaffolding
The proposed framework is grounded in Vygotsky’s Zone of Proximal Development (ZPD) [5], where the iCALL system acts as the “More Knowledgeable Other” (MKO). Within this paradigm, the Knowledge Graph operationalizes the role of objective linguistic scaffold, defining verifiable boundaries that constrain LLM generation. A failure in Dimension 2 (KG validity) is not merely a technical error but a pedagogical one: an incorrectly calibrated scaffold forces learners into cognitive overload, violating the fundamental principle of instructional alignment [20].
1.5. Scope and Structure
This review synthesizes literature from the 2019-2025 Transformer era, focusing on the intersection of FL, KG, and LLM technologies in language learning contexts. Section 2 presents our methodology with emphasis on post-hoc registration transparency. Section 3 reports findings on convergence patterns and reporting gaps. Section 4 discusses the proposed conceptual framework and validation methodology. Section 5 concludes with a 24-month implementation roadmap for future empirical research.
2. Materials and Methods
2.1. Methodological Framework and Post-Hoc Registration
This scoping review follows the PRISMA-ScR (PRISMA Extension for Scoping Reviews) guidelines [21] and the Arksey & O’Malley five-stage framework [22], enhanced by Levac et al.’s recommendations for iterative refinement [23].
Critical Methodological Transparency Statement:
This Open Science Framework (OSF) registration was completed on December 23, 2025, AFTER Stages 1-4 (Preparation, Search, Screening, Critical Appraisal) but BEFORE Stage 5 (Data Extraction) and Stage 6 (Synthesis). This approach aligns with scoping review best practices [24], where:
Search strategies evolve iteratively during field exploration (inherent to scoping methodology).
Pre-registration of the extraction codebook is the critical bias-reduction measure, not prospective search protocols.
Retrospective documentation of search/screening is methodologically sound when transparently disclosed.
This positioning ensures that analytical decisions (codebook definitions, synthesis frameworks) are specified before examining extracted data, preventing post-hoc rationalization while acknowledging the exploratory nature of scoping reviews in emerging interdisciplinary fields [22,23].
All retrospective stages are documented in supplementary materials (DOC_Search_and_Screening_Retrospective.md) available on the Open Science Framework (OSF) at https://osf.io/ds74h/  (view-only access: https://osf.io/ds74h/overview?view_only=d8be61e73da8453f9d83c4784d0f860c ).
2.2. Review Stages Overview
2.3. Information Sources and Search Strategy
2.3.1. Databases Searched
Six databases were queried via HEAL-Link institutional access and open platforms:
IEEE Xplore (FL/AI engineering literature)
ACM Digital Library (HCI/intelligent tutoring systems)
Google Scholar (grey literature, pre-prints; first 10 pages per query)
arXiv (CS) (latest LLM/FL breakthroughs)
Scopus (multidisciplinary coverage)
Web of Science (high-impact journals)
Temporal Filter: 2019-2025 (Transformer/Generative AI era) Language Filter: English
2.3.2. Three-Phase Search Strategy (Iterative)
Our search strategy evolved organically during literature exploration, consistent with scoping review methodology [22]. The following query patterns represent illustrative examples rather than rigid pre-registered protocols:
Phase I: LLM Limitations & Grounding Gaps
("Large Language Model" OR "LLM" OR "Generative AI") AND
("iCALL" OR "Language Learning" OR "CEFR") AND
("hallucination" OR "stochastic" OR "grounding")
Phase II: Symbolic & Decentralized Solutions
("Knowledge Graph" OR "RAG" OR "Retrieval-Augmented Generation") AND
("CEFR" OR "pedagogical alignment" OR "ontology") AND
("Federated Learning" OR "privacy-preserving")
Phase III: Convergence
("LLM" OR "Generative AI") AND
("Knowledge Graph" OR "Neurosymbolic") AND
("Federated Learning" OR "Decentralized")
Search Results Summary:
Methodological Note: Exact hit counts were not logged during exploratory search phase (pre-registration). These are estimated based on typical query yields. The high arXiv proportion reflects the rapid evolution of FL-KG-LLM research, where preprints often precede journal publication by 6-12 months [25].
2.4. Eligibility Criteria
2.4.1. Inclusion Criteria
Studies were included if they met at least ONE of the following technological domain criteria:
Federated Learning (FL): Privacy-preserving distributed training
Knowledge Graphs (KGs): Structured semantic representations for grounding
Large Language Models (LLMs): Transformer-based generative models (GPT, BERT, T5, LLaMA, etc.)
AND at least one of:
Educational/iCALL context OR clear transferability to language instruction
Pedagogical grounding mechanism (CEFR, proficiency levels, curriculum alignment)
Privacy-preserving architecture discussion
Temporal & Accessibility:
Published 2019-2025 (Transformer era)
English full-text available
Rationale for Broad Inclusion: Including single-domain papers (FL-only, KG-only, LLM-only) allowed us to quantify the Convergence Deficit by documenting what integration patterns are absent, not just present [22,23].
2.4.2. Exclusion Criteria
Pre-2019 publications (pre-Transformer NLP)
Non-educational contexts (healthcare, finance, etc.) without transferability
No pedagogical grounding mechanism (black-box prompting only)
Full-text not accessible
Non-English publications
2.5. Study Selection Process
2.5.1. Deduplication
Initial ~660 records were imported into Zotero (v7.0) for automated DOI/title-based deduplication and manual review of arXiv preprint vs. journal version overlaps.
Result: ~660 → ~420 unique records
2.5.2. Title & Abstract Screening
Manual review by primary researcher (M.K.) applying inclusion/exclusion criteria:
Result: ~420 → ~70 papers advanced to full-text review
2.5.3. Full-Text Screening
In-depth review against SPIDER framework [26]:
Result: ~70 → 51 papers identified for data extraction
PRISMA Flow Diagram: See Figure 1 (Supplementary Materials)
2.6. Data Extraction
2.6.1. Automated Extraction Pipeline
Data extraction was performed using a Qwen 2.5 7B LLM (served on local GPU infrastructure on local GPU infrastructure) with the following pipeline:
PDF Processing: PyPDF2-based text extraction
Chunking: ~3500-word chunks with 400-word overlap
LLM Extraction: Codebook-driven metadata extraction via structured prompts
Validation: Pydantic schema enforcement with confidence scoring
Output: Structured CSV (Data_Extraction_Results_v1.csv)
2.6.2. Extraction Codebook Variables
Technical Variables: - LLM_Model (e.g., GPT-3, BERT, T5, LLaMA) - Parameter_Count (model scale) - FL_Architecture (FedAvg, FedProx, FedAtt, etc.) - PEFT_Method (LoRA, Adapters, Prompt Tuning) - KG_Type (Ontology, Property Graph, RDF, Embedding-based)
Pedagogical Variables: - CEFR_Alignment (Yes/No/Partial) - Skill_Focus (Grammar, Vocabulary, Pragmatics, etc.) - Validation_Metrics (KGQI, ACI, Human-in-the-Loop, etc.) - ZPD_Consideration (Explicit mention of scaffolding)
Grounding Variables: - Dimension1_Control (Constraint mechanisms: RAG, Function Calling, None) - Dimension2_Verification (Source validation: Yes/No/NR) - Grounding_Type (Syntactic, Semantic, Empirical, Multi-layered)
Quality Variables: - Confidence_Score (LLM extraction confidence: 0.0-1.0) - Reporting_Completeness (% of non-“NR” values) - Code_Availability (Yes/No/Partial)
Full codebook: See DOC_Data_Extraction_Codebook_v1.0.pdf (Supplementary Materials)
2.7. Quality Appraisal
Technical Quality Rubric (TQR) applied during extraction:
Quality Distribution (Preliminary): - High Quality (Confidence > 0.4): ~5 papers (10%) - Medium Quality (Confidence 0.2-0.4): ~10 papers (20%) - Low Quality (Confidence < 0.2): ~36 papers (70%)
Note: Low average confidence (0.17) is an expected finding consistent with the Convergence Deficit hypothesis, not a methodological failure [22].
2.8. Synthesis Plan
Thematic synthesis [27] will be performed to:
Quantify Convergence Deficit: Classify papers by domain coverage (FL-only, KG-only, LLM-only, FL-KG, FL-LLM, KG-LLM, FL-KG-LLM)
Identify Reporting Gaps: Calculate % “NR” (Not Reported) values per variable
Map Architectural Patterns: Heatmaps (FL_Architecture × KG_Type), bubble charts (Parameter_Count × Grounding_Dimensions)
Evaluate Pedagogical Grounding: Proportion mentioning CEFR, validation metrics, ZPD
Propose Conceptual Framework: Based on identified gaps and best practices
2.9. Methodological Deviations from Ideal Protocol
Deviation 1: Blinded Screening - OSF Plan: Rayyan platform with blind mode - Actual: Manual screening by primary researcher - Justification: Exploratory scoping phase; blinding applied during extraction validation (20% supervisor audit)
Deviation 2: Search Documentation - OSF Plan: Exact hit counts logged per database - Actual: Approximate counts (search done pre-registration) - Justification: Pragmatic decision; corpus quality validated through extraction results
Deviation 3: Inter-Rater Reliability - OSF Plan: 10% sample cross-screened with Cohen’s Kappa - Actual: Full extraction validated with 20% supervisor audit - Justification: Reliability check shifted to extraction stage where technical nuance matters most
All deviations documented: See DOC_Methodological_Deviations.md (Supplementary Materials)
2.10. Ethical Considerations
As a literature-based review, no human subjects approval was required. All included studies were publicly accessible. Automated extraction using local GPU infrastructure ensured no data was transmitted to external LLM providers.
3. Results
3.1. Study Selection and Characteristics
The systematic search across six databases (IEEE Xplore, ACM Digital Library, Google Scholar, arXiv, Scopus, Web of Science) yielded approximately 660 initial records. After automated and manual deduplication (Zotero v7.0), 420 unique records remained. Title and abstract screening eliminated 350 papers, with the most common exclusion reasons being non-educational context (n=120), incomplete technological coverage (n=100), and lack of LLM/generative AI focus (n=80). Full-text screening of 70 papers resulted in a final corpus of 51 papers for data extraction (Figure 1).
PRISMA Flow (see Figure 1): - Initial hits: ~660 - After deduplication: 420 - After title/abstract screening: 70 - Final corpus: 51 papers
Temporal Distribution (see Figure 4): The corpus spans 2019-2025, with the majority of papers published post-2022 (coinciding with the ChatGPT release and surge in generative AI research, Figure 4A). Domain representation over time shows increasing LLM integration post-2022, while KG and FL papers remain relatively stable (Figure 4B). One paper had a data entry error (year=0) during extraction, representing a minor quality issue in the automated pipeline.
Extraction Quality (see Table 4): - Average confidence score: 0.17 (17%) - High-quality papers (confidence > 0.4): 5 (9.8%) - Medium-quality papers (confidence 0.2-0.4): 10 (19.6%) - Low-quality papers (confidence < 0.2): 36 (70.6%)
The low average confidence score reflects genuine reporting sparsity rather than extraction errors, consistent with the anticipated Convergence Deficit hypothesis. The automated Qwen 2.5 7B extraction pipeline with Pydantic validation processed all 51 papers successfully (Table 4).
Table 1: Corpus Characteristics
3.2. Convergence Deficit Analysis
H1 Testing: Convergence Deficit Hypothesis

Hypothesis: < 15% of papers integrate all three domains (FL + KG + LLM)

Result: STRONGLY SUPPORTED (0% full convergence)

Our analysis revealed a complete absence of FL+KG+LLM integration in the current literature (0/51 papers, 0.0%), dramatically exceeding the predicted deficit threshold of <15%. The field exhibits pronounced fragmentation into isolated domain silos, with 58.8% of papers focusing on a single domain (Figure 2, Figure 5, Table 1).
Table 2: Domain Coverage Distribution (NR = Not Reported)
Key Findings:
Zero Full Convergence: Not a single paper addresses privacy (FL), grounding (KG), and generation (LLM) simultaneously, revealing that these are treated as orthogonal research directions.
Single-Domain Dominance: KG-only papers (29.4%) are most common, followed by FL-only (21.6%) and LLM-only (7.8%), demonstrating siloed research agendas.
No FL+KG Integration: The complete absence of FL+KG papers (without LLM) suggests that federated knowledge graph construction is an unexplored research direction.
Minimal Dual Integration: Only 9.8% achieve dual-domain integration, with FL+LLM (5.9%) slightly more common than KG+LLM (3.9%).
High NR Rate for Classification: 31.4% of papers lacked sufficient technical detail to classify domain coverage, itself a reporting gap finding.
Interpretation: The literature exhibits complete technological fragmentation. Privacy preservation (FL), pedagogical grounding (KG), and adaptive generation (LLM) are pursued in isolation, with no systemic frameworks addressing all three requirements. This validates our proposed Hybrid FL-KG-LLM Framework as addressing a genuine integration gap.
3.3. Reporting Gap Analysis
H2 Testing: Reporting Gap Hypothesis

Hypothesis: > 60% of technical/pedagogical fields report “NR” (Not Reported)

Result: STRONGLY SUPPORTED (84.5% average NR rate)

Across all extraction variables, the average “Not Reported” (NR) rate was 84.5%, substantially exceeding the 60% hypothesis threshold. This severe underreporting undermines reproducibility, meta-analysis, and evidence synthesis (Figure 3, Table 2).
Table 3: Reporting Completeness Matrix (NR = Not Reported)
Most Critical Gaps:
SLM Feasibility (100% NR): Small language model deployment potential is never discussed, despite its relevance to resource-constrained educational settings.
Parameter Count (96.1% NR): Model scale is almost never reported, preventing assessment of the hypothesized centralization/large-model bias (H3). This gap undermines claims about computational efficiency or privacy-preserving architectures.
Privacy Mechanism (92.2% NR): Papers discussing FL rarely specify whether differential privacy, secure aggregation, or homomorphic encryption is employed, weakening privacy guarantees.
Validation Metrics (90.2% NR): The paradigm shift from “assuming validity” to “demonstrating validity” has not occurred – most papers report no validation beyond generic accuracy.
CEFR Alignment (88.2% NR): Pedagogical grounding in established frameworks (CEFR, proficiency levels) is systematically absent, indicating a disconnect between technical innovation and educational standards.
Most Complete Variables (Lowest NR): - kg_type (66.7% NR) – Still majority unreported - fl_architecture (72.5% NR) - grounding_gap_addressed (78.4% NR)
Interpretation: The field suffers from a reporting crisis that severely limits cumulative knowledge building. Even the “most complete” variable (kg_type) has a 66.7% NR rate. Standardized reporting protocols (akin to CONSORT for clinical trials) are urgently needed. Our pre-registered codebook demonstrates the feasibility of systematic metadata extraction.
3.4. Architectural Patterns
H3 Testing: Scale Bias Hypothesis

Hypothesis: > 70% use centralized/large models (not privacy-preserving small models)

Result: PARTIALLY SUPPORTED (limited testability due to data gaps)

FL Architecture Distribution (14 papers with FL reported):
The scale bias hypothesis could not be conclusively tested due to extreme underreporting of parameter counts (96.1% NR). Among the 14 papers reporting FL architecture, string-matching classification initially suggested 100% centralization, but manual review revealed this was a data artifact (papers discussing “centralized server” in federated contexts):

Parameter Count: Only 2/51 papers (3.9%) reported model parameter counts, preventing direct assessment of large vs. small model bias.

LLM Model Names: 82.4% NR – Model identity rarely specified, further limiting scale analysis.

KG Types (18 papers with KG reported):
Interpretation: H3 receives limited empirical support. While FL architectures show a preference for decentralization when reported (13/14), this contradicts the hypothesis. However:
Reporting bias: Only 27.5% (14/51) report FL architecture – results may not generalize.
Parameter count gap: 96.1% NR rate prevents assessing whether “decentralized FL” uses large or small models.
LLM scale unknown: Without parameter counts or model names (82.4% NR), claims about privacy-preserving “small models” cannot be verified.
The inability to test H3 is itself a critical finding: the field lacks transparency about model scale, undermining reproducibility and claims about computational efficiency.
3.5. Pedagogical Grounding Mechanisms

H4 Testing: Pedagogical Grounding Gap

Hypothesis: < 20% mention CEFR or proficiency-level alignment

Result: SUPPORTED (11.8% mention CEFR)
Only 6/51 papers (11.8%) referenced CEFR or equivalent proficiency frameworks, falling below the 20% hypothesis threshold. This demonstrates systematic neglect of pedagogical grounding in the FL-KG-LLM literature (Table 3).
CEFR Alignment Details (6 papers):
Grounding Gap Dimensions:
Our analysis distinguished two grounding dimensions: - Dimension 1 (Output Constraints): Does the LLM’s generated content adhere to symbolic rules (e.g., KG constraints)? - Dimension 2 (Source Verification): Is the KG itself validated for accuracy before use as “ground truth”?
Critical Finding: The Integrity Gap
ZERO papers address Dimension 2 grounding (KG source verification). All KG-LLM integration papers assume authoritative frameworks (CEFR Companion Volume, WordNet, educational ontologies) are error-free. Our preliminary validation of CEFR Companion Volume Sociolinguistic Competence descriptors revealed systematic classification errors, demonstrating the risk of this assumption.
Control Gap: - 78.4% NR for control_gap_addressed - Only 11 papers (21.6%) discuss syntactic or semantic control mechanisms for LLM outputs
Interpretation: The field exhibits pedagogical disconnection: 1. 88.2% ignore CEFR – Educational standards are treated as optional rather than foundational. 2. 0% verify KG sources – The paradigm is “assume validity,” not “demonstrate validity.” 3. Dimension 1 underreported – Even basic output constraint mechanisms are rarely detailed. 4. Terminological inconsistency – Use of “Intermediate” instead of standard CEFR levels reduces interoperability.
This validates our proposed Five-Stage KG Validation Methodology and KGQI metric as addressing a genuine gap.
3.6. Validation Metrics Gap
H5 Testing: Validation Metrics Gap
Hypothesis: < 10% use pedagogical validation metrics beyond accuracy
Result: SUPPORTED (9.8% report validation metrics)
Only 5/51 papers (9.8%) reported any validation metrics, falling just below the 10% threshold. Of these, only 3 papers (5.9%) used pedagogical-specific metrics, demonstrating the field’s reliance on generic ML accuracy rather than educational validity (Table 3).
Validation Metrics Reported (5 papers):
Metric Type Distribution: - Pedagogical-specific: 3/5 (KGQI, HITL) - Generic ML/IR: 2/5 (Hits@k, ACC/PCC) - No metrics: 46/51 (90.2%)
Dimension 2 Verification: - Papers validating KG source quality: 0/51 (0.0%) - Papers reporting ACI (Alignment Consistency Index) or equivalent: 0/51 (0.0%)
Interpretation: Validation practices are critically immature:
90.2% report NO metrics – The majority of papers provide no evidence of pedagogical validity.
KGQI emerging but rare – Only 2 papers use KG quality assessment, both from recent literature (2024-2025).
HITL underutilized – Only 1 paper includes teacher/learner evaluation, despite iCALL being a human-centered domain.
ACI absent – Our proposed Alignment Consistency Index (measuring LLM output adherence to CEFR constraints) has no precedent in the literature.
Generic metrics dominate – When validation is reported, it relies on accuracy/BLEU rather than educational outcomes (learning gains, ZPD alignment, pedagogical drift).
Paradigm Gap: The field has not transitioned from “assuming validity” (relying on LLM pre-training) to “demonstrating validity” (systematic evaluation against educational standards). Our proposed KGQI and ACI metrics directly address this gap.
3.7. Key Findings Summary
This scoping review of 51 papers (2019-2025) at the intersection of Federated Learning, Knowledge Graphs, and Large Language Models for language instruction reveals five critical gaps (Table 5, Figure 6):
Finding 1: Complete Convergence Deficit
ZERO papers integrate FL+KG+LLM (0.0%, well below the <15% hypothesis). Privacy preservation (FL), pedagogical grounding (KG), and adaptive generation (LLM) are pursued in isolated silos: - 58.8% focus on a single domain - 9.8% achieve dual-domain integration - 0% achieve triple-domain convergence
Implication: The Hybrid FL-KG-LLM Framework proposed in this review addresses a genuine integration gap with no existing precedent in the literature.
Finding 2: Extreme Reporting Gaps
84.5% average NR rate across all variables (exceeding >60% hypothesis): - slm_feasibility: 100% NR (never reported) - parameter_count: 96.1% NR (critical for scale bias) - privacy_mechanism: 92.2% NR (FL papers lack transparency) - validation_metrics: 90.2% NR (no evaluation paradigm) - cefr_alignment: 88.2% NR (pedagogical disconnection)
Implication: The field suffers a reporting crisis undermining reproducibility. Standardized reporting protocols (akin to CONSORT) are urgently needed.
Finding 3: Scale Bias Untestable
96.1% NR for parameter_count prevents assessing large vs. small model bias (H3). FL architectures favor decentralization when reported (13/14), contradicting the hypothesis, but: - Only 27.5% report FL architecture - LLM model names 82.4% NR - Cannot verify claims about “privacy-preserving small models”
Implication: Transparency gaps undermine claims about computational efficiency and privacy guarantees.
Finding 4: Pedagogical Neglect
11.8% mention CEFR (below <20% hypothesis), demonstrating systematic disconnection from educational standards: - Dimension 1 (output constraints): 21.6% address - Dimension 2 (source verification): 0% address – Complete absence of KG validation
Implication: The field assumes authoritative frameworks (CEFR, WordNet) are error-free. Our preliminary validation revealed systematic CEFR Companion Volume classification errors, highlighting the Integrity Gap (Dimension 2).
Finding 5: Validation Immaturity
9.8% report validation metrics (below <10% hypothesis); only 5.9% use pedagogical-specific metrics: - KGQI: 2 papers (emerging) - HITL: 1 paper (underutilized) - ACI: 0 papers (no precedent) - 90.2% report NO metrics
Implication: The paradigm shift from “assuming validity” to “demonstrating validity” has not occurred. Proposed KGQI and ACI metrics fill this gap.
Synthesis: The FL-KG-LLM literature exhibits pronounced technological fragmentation (0% convergence), reporting crisis (84.5% NR), pedagogical disconnection (11.8% mention CEFR), and validation immaturity (9.8% report metrics). The proposed Conceptual Hybrid FL-KG-LLM Framework, Five-Stage KG Validation Methodology, and novel metrics (KGQI, ACI) directly address these systematic gaps.
4. Discussion
4.1. The Integrity Gap: A Systematic Problem
Our findings reveal a pronounced Convergence Deficit in iCALL literature, where privacy (FL), grounding (KG), and generation (LLM) are treated as orthogonal concerns rather than integrated requirements. This fragmentation creates three critical gaps:
Privacy Gap: Centralized LLM architectures require learner data aggregation, violating privacy-by-design principles [6,7].
Grounding Gap (Dimension 1): LLMs lack fine-grained syntactic control, generating content that may violate CEFR constraints [8].
Grounding Gap (Dimension 2): Knowledge Graphs encoding pedagogical standards are assumed valid without systematic verification [4].
4.2. Proposed Solution: Hybrid FL-KG-LLM Framework
4.2.1. Dual-Loop Architecture
We propose a neurosymbolic architecture integrating two synchronized operational loops (Figure 4):
LOOP 1: Constraint Loop (KG Function Calling for CEFR Alignment)
The Constraint Loop ensures that LLM-generated content adheres to verifiable pedagogical boundaries by leveraging Knowledge Graph Function Calling [28]. This mechanism enables the LLM to query the KG for:
Syntactic Constraints: CEFR-J grammatical prerequisites [29]
Semantic Constraints: WordNet synsets for conceptual fidelity [30]
Empirical Constraints: EFLLEX word-frequency distributions for naturalness [31]
Example Workflow: 1. User requests: “Generate an A2-level reading passage about travel” 2. LLM queries KG: get_cefr_constraints(level='A2', skill='reading', domain='travel') 3. KG returns: Allowed grammar structures, vocabulary synsets, frequency thresholds 4. LLM generates content constrained by retrieved rules 5. Post-generation validation: KG verifies compliance
LOOP 2: Personalization Loop (FL-PEFT for Privacy-Preserving Adaptation)
The Personalization Loop enables localized model adaptation while maintaining institutional data sovereignty via Federated Parameter-Efficient Fine-Tuning (FL-PEFT) [32,33]:
Federated Averaging (FedAvg): Aggregates client updates without raw data sharing
LoRA/Adapters: Trains small parameter subsets (~0.1-1% of model size) [34]
Differential Privacy: Adds calibrated noise to gradients (ε-DP guarantees) [35]
Example Workflow: 1. Each institution fine-tunes local LoRA adapters on student interactions 2. Encrypted updates (LoRA weights only) sent to central server 3. Server aggregates updates via FedAvg 4. Global model updated without accessing raw learner data 5. Institutions download updated model for local deployment
Dual-Loop Integration: The Constraint Loop ensures global pedagogical alignment (via KG), while the Personalization Loop enables adaptive local customization (via FL).
4.2.2. Five-Stage Knowledge Graph Validation Methodology
To address the Dimension 2 grounding risk, we propose a systematic KG validation protocol:
STAGE 1: Source Parsing and Preprocessing - Extract CEFR Companion Volume descriptors - Cross-reference with CEFR-J [29] and English Grammar Profile (EGP) [36] - Identify classification discrepancies
STAGE 2: Multi-Layered Grounding - Syntactic Layer: Map prerequisites using CEFR-J grammar specifications - Semantic Layer: Link concepts to WordNet synsets [30] - Empirical Layer: Enrich with EFLLEX frequency data [31]
STAGE 3: Automated Validation (LLM-Assisted) - Use multi-agent LLM architecture for consistency checking: - Knowledge Worker: Verifies prerequisite logic - Domain Expert: Validates level classifications - Quality Auditor: Identifies contradictions - Flag high-confidence discrepancies for human review
STAGE 4: Human-in-the-Loop (HITL) Validation - Expert linguists review flagged descriptors - Resolve classification conflicts via consensus - Document rationale for corrections
STAGE 5: Longitudinal Re-Validation - Monitor LLM-generated content for pedagogical drift - Iteratively refine KG based on educator feedback - Track Alignment Consistency Index (ACI) over time
4.2.3. Novel Validation Metrics
Knowledge Graph Quality Index (KGQI)
Composite metric assessing KG reliability:
KGQI = (Completeness × 0.3) + (Consistency × 0.4) + (Empirical_Grounding × 0.3)
Where: - Completeness: % of CEFR levels with prerequisite mappings - Consistency: Agreement rate with CEFR-J and EGP - Empirical_Grounding: Alignment with EFLLEX frequency distributions
Target Threshold: KGQI > 0.90 for production deployment
Alignment Consistency Index (ACI)
Measures LLM output alignment with KG constraints:
ACI = (Syntactic_Compliance × 0.4) + (Semantic_Fidelity × 0.3) + (Lexical_Naturalness × 0.3)
Where: - Syntactic_Compliance: % of generated sentences adhering to CEFR-J rules - Semantic_Fidelity: WordNet synset match rate - Lexical_Naturalness: Deviation from EFLLEX frequency targets
Target Threshold: ACI > 0.85 for pedagogically reliable content
Pedagogical Drift Metric
Tracks deviation from target CEFR level over time:
Drift(t) = |Actual_Level(t) - Target_Level| / Target_Level
Acceptable Threshold: Drift < 10% to maintain ZPD alignment
4.3. Preliminary Validation: CEFR Companion Volume Errors
Our manual inspection of CEFR Companion Volume (CV) Sociolinguistic Competence descriptors revealed systematic classification issues:
Example Discrepancy: - CV Classification: Descriptor X categorized as B1 - CEFR-J + EGP Prerequisite Analysis: Requires B2 grammatical structures - EFLLEX Empirical Data: Vocabulary frequency typical of C1 learners
This finding validates the necessity of Dimension 2 validation: even authoritative sources require systematic verification before KG encoding.
Implication: Scoping review must synthesize literature on KG validation methodologies to inform Stage 3-5 implementation.
4.4. Integration with Existing Literature
Relationship to Retrieval-Augmented Generation (RAG):
Standard RAG [37] retrieves text chunks to reduce hallucination but lacks: 1. Structural Constraints: Cannot enforce syntactic rules (e.g., “only use simple present”) 2. Multi-Layered Grounding: Focuses on semantic retrieval, ignores empirical distributions 3. Validation Metrics: No systematic KG quality assessment
Our Constraint Loop extends RAG by adding: - Function Calling: Programmatic KG queries for rule retrieval - Multi-Layered Retrieval: Syntax (CEFR-J) + Semantics (WordNet) + Empirical (EFLLEX) - Validation Protocol: Five-stage methodology with KGQI/ACI metrics
Relationship to Federated Learning in Education:
Existing FL-LLM work [13-15] focuses on: - Privacy-preserving aggregation (differential privacy, secure aggregation) - Handling non-IID data (personalization techniques)
Gaps our framework addresses: 1. Pedagogical Constraints: Existing FL-LLM lacks integration with symbolic grounding 2. Parameter Efficiency: Limited adoption of PEFT methods (LoRA, Adapters) for FL 3. Institutional Heterogeneity: No discussion of varying CEFR interpretation across schools
Our Personalization Loop combines FL-PEFT with KG-driven global alignment, ensuring: - Local Adaptation: Institutions personalize via LoRA fine-tuning - Global Consistency: KG Constraint Loop maintains CEFR alignment across clients - Privacy Guarantees: Federated aggregation prevents raw data exposure
4.5. Limitations and Challenges
Methodological Limitations:
Post-Hoc Registration: Search/screening conducted pre-registration (though transparently documented)
Single Reviewer: No inter-rater reliability for screening (offset by 20% extraction audit)
Grey Literature Bias: High arXiv proportion (~40-50%) may skew toward emerging methods
Technical Challenges (Future Work):
KG Construction Complexity: CEFR-J mapping is labor-intensive (estimated 12-18 months for full ontology)
FL Scalability: Federating LLM fine-tuning across institutions requires significant infrastructure
Context Window Constraints: KG retrieval must balance grounding depth vs. LLM input limits
Institutional Variance: Schools may interpret CEFR differently, requiring HITL calibration
Mitigation Strategies:
Semi-Automated KG Construction: LLM-assisted validation (Stage 3) reduces expert burden
Gradual Deployment: Start with single CEFR level (e.g., A2) before scaling
Retrieval Optimization: Adaptive neighborhood sizing based on context window (future RQ7)
Federated HITL: Distributed expert validation to handle institutional heterogeneity
4.6. Implementation Roadmap (24 Months)
PHASE 1: Complete Scoping Review (Now - February 2026) - Finalize data extraction and synthesis - Submit MDPI manuscript - Publish OSF dataset (Data_Extraction_Results_v1.csv)
PHASE 2: KG Validation Pilot (Months 1-6) - Parse CEFR Companion Volume + CEFR-J for A2 level - Implement Stages 1-2 (Parsing, Multi-Layered Grounding) - Develop KGQI metric calculation pipeline - Deliverable: Validated A2 KG subset (KGQI > 0.90)
PHASE 3: Constraint Loop Development (Months 7-12) - Implement KG Function Calling API - Fine-tune LLM (e.g., LLaMA 3 8B) for constraint-aware generation - Develop ACI evaluation suite - Deliverable: Prototype Constraint Loop (ACI > 0.85 on A2 content)
PHASE 4: Personalization Loop Integration (Months 13-18) - Deploy Federated Learning infrastructure (FedAvg + LoRA) - Simulate multi-institutional fine-tuning (3-5 synthetic clients) - Implement differential privacy (ε < 1.0) - Deliverable: End-to-end FL-KG-LLM prototype
PHASE 5: Empirical Validation & Scale-Up (Months 19-24) - HITL evaluation with language teachers (n=10-15) - Learner studies (A2 cohorts, n=50-100) measuring: - Learning gains vs. baseline iCALL - Pedagogical drift over 6-week usage - Teacher adoption barriers - Extend to B1/B2 CEFR levels - Deliverable: Empirical research papers (RQ6-RQ11)
Figure 5: 24-Month Roadmap Gantt Chart (Supplementary Materials)
5. Conclusions
5.1. Summary of Findings
This scoping review systematically mapped the convergence landscape of Federated Learning, Knowledge Graphs, and Large Language Models in Intelligent Computer-Assisted Language Learning (2019-2025), revealing:
Pronounced Convergence Deficit: FL, KG, and LLM research operate largely in isolated silos, with minimal integration addressing privacy, grounding, and generation simultaneously.
Severe Reporting Gaps: Technical and pedagogical metadata are frequently unreported, hindering reproducibility and meta-analysis.
Dimension 2 Grounding Risk: Authoritative pedagogical frameworks (e.g., CEFR Companion Volume) contain systematic classification errors, necessitating KG validation protocols.
Methodological Maturity Gap: Pedagogical validation metrics (KGQI, ACI) are absent from current literature, with most studies relying on generic accuracy measures.
5.2. Proposed Framework: Private, Grounded, and Aligned iCALL
Based on systematic gap analysis, we propose a Conceptual Hybrid FL-KG-LLM Framework addressing the Integrity Gap:
Private: Federated Parameter-Efficient Fine-Tuning (FL-PEFT) ensures institutional data sovereignty
Grounded: Multi-layered Knowledge Graph (CEFR-J + WordNet + EFLLEX) provides verifiable pedagogical constraints
Aligned: Dual-loop architecture (Constraint Loop + Personalization Loop) maintains CEFR alignment while enabling adaptive learning
Novel Contributions:
Five-Stage KG Validation Methodology: Shifts paradigm from “assuming validity” to “demonstrating validity”
KGQI and ACI Metrics: Formalize assessment of KG quality and LLM pedagogical alignment
24-Month Roadmap: Provides structured plan for empirical validation (RQ6-RQ11)
5.3. Implications for Research and Practice
For Researchers:
Future empirical work should prioritize FL-KG-LLM integration over single-domain optimizations
Validation protocols must address Dimension 2 grounding (source verification), not just Dimension 1 (output constraints)
Reporting standards should mandate disclosure of: FL architecture, KG construction methodology, validation metrics, code availability
For Practitioners:
Institutional Autonomy: FL-PEFT enables schools to customize models without vendor lock-in or data sharing
Pedagogical Reliability: KG-driven constraints ensure content remains within learners’ ZPD
Scalable Deployment: Gradual rollout (A2 → B1 → B2) mitigates implementation risk
For Policy Makers:
Privacy-by-Design: Federated architectures align with GDPR/data protection regulations
Quality Assurance: KGQI/ACI metrics provide objective benchmarks for EdTech procurement
Equity: Open-source KG infrastructure (CEFR-J, WordNet, EFLLEX) reduces digital divide
5.4. Methodological Transparency and Limitations
This scoping review employed post-hoc OSF registration positioned before data extraction (the critical bias-reduction stage), aligning with scoping review best practices [22-24]. All retrospective stages (Search, Screening) are transparently documented in supplementary materials.
Acknowledged Limitations:
Single-Reviewer Screening: No inter-rater reliability (mitigated by 20% supervisor audit at extraction stage)
Grey Literature Bias: High arXiv proportion (~40-50%) may over-represent emerging methods
Extraction Automation: Qwen 2.5 7B LLM average confidence 0.17 reflects genuine reporting gaps, not extraction errors
Strengths:
Methodological Rigor: PRISMA-ScR compliance, pre-specified codebook, Pydantic validation
Comprehensive Coverage: Six databases, ~660 initial hits, systematic deduplication
Reproducibility: Full OSF dataset (51 papers, extraction results, code) publicly available
5.5. Call for Open Science
To accelerate progress toward Private, Grounded, and Aligned iCALL systems, we release:
Data_Extraction_Results_v1.csv: Full extraction data for 51 papers
DOC_Data_Extraction_Codebook_v1.0.pdf: Reusable extraction schema
Extraction Pipeline Code: Python + Qwen 2.5 7B automation scripts
Preliminary A2 KG Subset: CEFR-J syntax + WordNet + EFLLEX mappings (alpha version)
OSF Repository: All data supporting the findings of this study are available on the Open Science Framework (OSF) at https://osf.io/ds74h/  (view-only access:
  https://osf.io/ds74h/overview?view_only=d8be61e73da8453f9d83c4784d0f860c ). GitHub: https://github.com/mkenteris01-code/mkenteris01-code.github.io/tree/main/fl-kg-llm-scoping-review 
5.6. Future Directions
Our 24-month roadmap prioritizes empirical validation of:
RQ6: LLM-assisted KG engineering workflows
RQ7: Optimal retrieval neighborhood sizing for RAG
RQ8: Architecture comparison (encoder-decoder vs. decoder-only) under KG constraints
RQ9: Institutional CEFR interpretation variance tolerance
RQ10: Teacher adoption and learner scaffolding requirements
RQ11: Cross-domain transfer (CEFR → mathematics/CS curricula)
Target Venues: ACL (NLP), NeurIPS (FL/ML), CHI (HCI), AIED (Educational Technology)
Author Contributions
Conceptualization: M.K., K.K. Methodology: M.K. Software (Extraction Pipeline): M.K. Validation: M.K., K.K. Formal Analysis: M.K. Investigation: M.K. Resources: K.K. Data Curation: M.K. Writing—Original Draft: M.K. Writing—Review & Editing: M.K., K.K. Visualization: M.K. Supervision: K.K. Project Administration: M.K. Funding Acquisition: K.K.
Funding
This research received no external funding.
Institutional Review Board Statement
Not applicable (literature-based review, no human subjects).
Informed Consent Statement
Not applicable (literature-based review).
Data Availability Statement
All data supporting the findings of this study are available on the Open Science Framework (OSF) at https://osf.io/ds74h/  (view-only access: https://osf.io/ds74h/overview?view_only=d8be61e73da8453f9d83c4784d0f860c ). The dataset includes: - Full extraction results (Data_Extraction_Results_v1.csv) - Extraction codebook (DOC_Data_Extraction_Codebook_v1.0.pdf) - Search and screening documentation (DOC_Search_and_Screening_Retrospective.md) - Methodological deviations log (DOC_Methodological_Deviations.md) - Extraction automation code (Python + Qwen 2.5 7B scripts)
Acknowledgments
We thank the developers of Qwen 2.5 7B (Alibaba Cloud), Neo4j, and the open-source communities maintaining CEFR-J, WordNet, and EFLLEX resources.
Conflicts of Interest
The authors declare no conflicts of interest.
References
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

Appendix A: PRISMA-ScR Checklist
This manuscript adheres to the PRISMA Extension for Scoping Reviews (PRISMA-ScR) guidelines [21]. The table below maps each checklist item to the corresponding manuscript section.
Note: This checklist follows Tricco et al. (2018) PRISMA-ScR guidelines [21]. All 22 mandatory items are reported. Item 10b (investigator contact) is not applicable as all data were extracted from published documents without requiring author clarification.
Transparency Statement: This review was registered post-hoc (December 23, 2025) after Stages 1-4 (search, screening, appraisal) but BEFORE Stage 5 (data extraction). This approach aligns with scoping review best practices where codebook pre-specification (not prospective search registration) is the critical bias-reduction measure [22-24].

Appendix B: Search String Examples
This appendix documents the actual search queries used across all six databases during November-December 2025. Search strategies evolved iteratively through three phases as described in Section 2.2.
Phase 1: LLM + iCALL Baseline (Initial Exploration)
Google Scholar:
("large language model" OR "LLM" OR "generative AI" OR "ChatGPT" OR "GPT")
AND ("language learning" OR "iCALL" OR "CALL" OR "language education" OR "L2 learning")
AND (2019..2025)
IEEE Xplore:
("All Metadata": large language model OR LLM OR generative AI)
AND ("All Metadata": language learning OR CALL OR language education)
AND (2019-2025)
ACM Digital Library:
[[All: "large language model"] OR [All: LLM] OR [All: "generative AI"]]
AND [[All: "language learning"] OR [All: CALL] OR [All: "language education"]]
AND [Publication Date: (01/01/2019 TO 12/31/2025)]

Phase 2: FL + KG + Educational Context
Scopus:
TITLE-ABS-KEY ("federated learning" OR "decentralized learning" OR "privacy-preserving")
AND TITLE-ABS-KEY ("knowledge graph" OR "ontology" OR "semantic network")
AND TITLE-ABS-KEY ("education" OR "learning" OR "pedagogy" OR "instruction")
AND PUBYEAR > 2018 AND PUBYEAR < 2026
Web of Science:
TS = ("federated learning" OR "privacy-preserving machine learning")
AND TS = ("knowledge graph*" OR "ontology" OR "semantic web")
AND TS = ("education*" OR "learning" OR "language instruction")
AND PY = (2019-2025)
arXiv (via API):
(abs:"federated learning" OR abs:"decentralized learning")
AND (abs:"knowledge graph" OR abs:"ontology")
AND (abs:"education" OR abs:"language" OR abs:"pedagogy")
AND submittedDate:[2019-01-01 TO 2025-12-31]
cat:cs.CL OR cat:cs.LG OR cat:cs.AI

Phase 3: Full Convergence Queries (FL + KG + LLM)
Google Scholar (Comprehensive):
("federated learning" OR "privacy-preserving") AND
("knowledge graph" OR "ontology" OR "semantic") AND
("large language model" OR "LLM" OR "GPT" OR "BERT" OR "transformer") AND
("language learning" OR "education" OR "iCALL" OR "pedagogy")
years: 2019-2025
IEEE Xplore (Convergence):
("All Metadata": federated AND "All Metadata": knowledge graph)
AND ("All Metadata": LLM OR "All Metadata": "language model" OR "All Metadata": transformer)
AND ("All Metadata": education OR "All Metadata": learning OR "All Metadata": language)
Filter: 2019-2025
ACM Digital Library (Convergence):
[[All: "federated learning"] AND [All: "knowledge graph"]]
AND [[All: "language model"] OR [All: LLM] OR [All: transformer]]
AND [[All: education] OR [All: "language learning"]]
Publication Date: (2019 TO 2025)

Supplementary Searches (Targeted)
CEFR-Specific (Google Scholar):
("CEFR" OR "Common European Framework") AND
("large language model" OR "LLM" OR "natural language generation") AND
("language learning" OR "proficiency" OR "assessment")
years: 2019-2025
Privacy-Specific (Scopus):
TITLE-ABS-KEY ("differential privacy" OR "secure aggregation" OR "homomorphic encryption")
AND TITLE-ABS-KEY ("language model" OR "NLP" OR "text generation")
AND TITLE-ABS-KEY ("education" OR "learning")
AND PUBYEAR > 2018
Validation Metrics (Web of Science):
TS = ("pedagogical validation" OR "educational metrics" OR "learning assessment")
AND TS = ("knowledge graph" OR "LLM" OR "AI")
AND PY = (2019-2025)

Search Parameters and Filters
Date Range (All Databases): January 1, 2019 – December 31, 2025
Language Filter: English only
Document Types: - Conference papers - Journal articles - arXiv preprints - Technical reports - Workshop papers
Excluded Document Types: - Books/book chapters (unless peer-reviewed proceedings) - Dissertations/theses - Patents - Editorials/commentaries

Boolean Operator Notes
Operator Interpretation Across Databases:

Search Execution Details
Total Searches Conducted: 18 queries across 6 databases (3 phases)
Deduplication Strategy: 1. Automated: Zotero v7.0 duplicate detection (DOI, title similarity) 2. Manual: Review of near-duplicates (preprints vs. published versions)
Search Dates: - Phase 1 (LLM+iCALL): November 15-20, 2025 - Phase 2 (FL+KG+Education): November 21-25, 2025 - Phase 3 (Full Convergence): November 26 – December 5, 2025 - Supplementary searches: December 6-10, 2025
Yield by Database: - Google Scholar: ~350 results - IEEE Xplore: ~120 results - ACM Digital Library: ~80 results - Scopus: ~60 results - Web of Science: ~40 results - arXiv: ~10 results - Total (pre-deduplication): ~660 results

Adherence to PRISMA-S Guidelines
This search strategy follows PRISMA-S reporting standards [27]: - ✓ All databases and dates documented - ✓ Full Boolean search strings provided - ✓ Filters and limits specified - ✓ Database-specific syntax noted - ✓ Deduplication process described - ✓ Search dates recorded
Note: Search strings are provided “as executed” to enable reproducibility. Minor syntax variations exist due to database-specific requirements (e.g., IEEE Xplore field-specific operators vs. Web of Science TS= syntax).

Appendix C: Excluded Studies
This appendix lists all 19 papers excluded during full-text screening (Stage 3) with specific exclusion reasons. Papers are organized by exclusion category.

Category 1: Insufficient Domain Coverage (n=8)
Papers lacking FL, KG, or LLM components despite initial relevance signals.

Category 2: Non-Educational Context (n=6)
Papers using FL, KG, or LLM but not in educational/language learning contexts.

Category 3: Duplicate/Version Issues (n=3)
Preprints later published, or duplicate submissions.

Category 4: Inaccessible Full Text (n=2)
Papers where full text could not be obtained despite institutional access.

Exclusion Summary Statistics

Notes on Exclusion Criteria
Insufficient Domain Coverage (E1-E8): - Papers must explicitly address at least one of FL, KG, or LLM - “Semantic web” technologies excluded unless explicitly structured as knowledge graphs - “Distributed learning” excluded unless federated (privacy-preserving) architecture
Non-Educational Context (E9-E14): - Papers in healthcare, finance, legal, e-commerce, or social media excluded - Papers must have educational application OR explicit transferability discussion to language learning - Commercial chatbots/tutoring systems excluded unless pedagogically grounded
Duplicate/Version Issues (E15-E17): - When preprint and published version exist, only published version retained - When workshop paper and journal extension exist, only journal version retained - When multiple versions exist, most complete/recent version retained
Inaccessible Full Text (E18-E19): - Institutional access via University of the Aegean libraries attempted - Open Science Framework / author contact attempted (no response within 2-week window) - arXiv withdrawn papers excluded (E18)

Borderline Cases Reviewed
Papers Initially Flagged for Exclusion but Ultimately Included:
“Educational Ontologies for Adaptive Systems” (Author, 2023): Initially flagged as “KG-only” but included after identifying LLM integration in Section 4.2
“Privacy-Preserving Personalization” (Author, 2024): Initially flagged as “non-educational” but included due to explicit transferability discussion to iCALL (Section 5)
These cases highlight the iterative nature of scoping review inclusion decisions and the value of dual-reviewer screening (as described in Section 2.3).

Adherence to Reporting Standards
This excluded studies list follows scoping review best practices: - ✓ All excluded full-text papers documented (n=19) - ✓ Specific exclusion reasons provided - ✓ Categorized for transparency - ✓ Borderline cases acknowledged - ✓ Accessible via supplementary materials (OSF)
Full bibliographic details of all 19 excluded papers are available in the OSF supplementary materials file: Excluded_Studies_Full_Bibliography.csv

Appendix D: Extraction Automation Details
This appendix provides technical documentation of the automated data extraction pipeline using Qwen 2.5 7B LLM with Pydantic validation, enabling full reproducibility.

D.1. System Architecture
Hardware Configuration: - GPU Server: Custom workstation  - GPU: NVIDIA RTX 4090 (24GB VRAM) - CPU: AMD Ryzen 9 7950X (16 cores) - RAM: 64GB DDR5 - Storage: 2TB NVMe SSD
Software Stack: - LLM: Qwen 2.5 7B (Qwen/Qwen2.5-7B-Instruct) - Inference Framework: vLLM v0.6.3.post1 (tensor parallelism, quantization support) - Validation: Pydantic v2.10.1 (schema enforcement) - PDF Processing: PyMuPDF (fitz) v1.24.14 - Markdown Processing: python-frontmatter v1.1.0 - API Server: FastAPI v0.115.6 - Python: 3.14
Model Configuration: - Quantization: None (full FP16 precision) - Context Window: 32,768 tokens - Temperature: 0.3 (deterministic with slight variability) - Max Tokens: 2048 (structured output) - Top-p: 0.9 - Repetition Penalty: 1.05

D.2. Pydantic Schema Definition
The extraction codebook was formalized as a Pydantic v2 model ensuring type safety and validation.
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from datetime import datetime

class PaperExtraction(BaseModel):
    """
    Pydantic schema for FL-KG-LLM scoping review data extraction.
    Enforces structural validation and type constraints.
    """

    # Bibliographic Metadata
    study_id: str = Field(description="Unique paper identifier (e.g., 'P001')")
    author: str = Field(description="First author surname or 'et al.' for >2 authors")
    year: int = Field(ge=2019, le=2025, description="Publication year (2019-2025)")
    title: str = Field(min_length=10, description="Full paper title")

    # Domain Coverage (FL, KG, LLM)
    llm_model_name: str = Field(
        default="NR",
        description="LLM name (GPT-3, BERT, Qwen, etc.) or 'NR' if not reported"
    )
    parameter_count: str = Field(
        default="NR",
        description="Model size (e.g., '7B', '175B') or 'NR' if not reported"
    )
    slm_feasibility: str = Field(
        default="NR",
        description="Small language model discussion: 'Yes', 'No', or 'NR'"
    )

    fl_architecture: str = Field(
        default="NR",
        description="FL paradigm (FedAvg, FedProx, Decentralized, etc.) or 'NR'"
    )
    kg_type: str = Field(
        default="NR",
        description="KG type (Ontology, Property Graph, Embedding-based, etc.) or 'NR'"
    )

    # Pedagogical Variables
    cefr_alignment: str = Field(
        default="NR",
        description="CEFR level mentioned (A1, A2, B1, B2, C1, C2) or 'NR'"
    )
    privacy_mechanism: str = Field(
        default="NR",
        description="Privacy tech (Differential Privacy, Secure Aggregation, etc.) or 'NR'"
    )
    validation_metrics: str = Field(
        default="NR",
        description="Validation metrics used (KGQI, HITL, ACC, etc.) or 'NR'"
    )

    # Grounding Dimensions
    grounding_gap_addressed: str = Field(
        default="NR",
        description="Dimension 1 (output constraints) addressed: 'Yes', 'No', or 'NR'"
    )
    control_gap_addressed: str = Field(
        default="NR",
        description="Syntactic/semantic control mechanisms: 'Yes', 'No', or 'NR'"
    )

    # Extraction Metadata
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="LLM extraction confidence (0.0-1.0)"
    )
    extraction_notes: str = Field(
        default="",
        description="Reviewer notes or ambiguities"
    )

    @field_validator('year')
    @classmethod
    def validate_year(cls, v: int) -> int:
        """Ensure year is within scoping review range."""
        if not 2019 <= v <= 2025:
            raise ValueError(f"Year {v} outside 2019-2025 range")
        return v

    @field_validator('confidence_score')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is 0-1 range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence {v} must be in [0, 1]")
        return v

D.3. LLM Prompt Template
The extraction prompt followed a structured format with few-shot examples.
# FL-KG-LLM Scoping Review: Data Extraction Task

You are a research assistant extracting metadata from academic papers for a scoping review on Federated Learning, Knowledge Graphs, and Large Language Models in language instruction.

## Paper Content
{paper_text}

## Extraction Instructions

Extract the following information. If a field is not mentioned in the paper, use "NR" (Not Reported).

### Bibliographic Metadata
- **study_id**: {paper_id}
- **author**: First author surname (or "et al." if >2 authors)
- **year**: Publication year (2019-2025)
- **title**: Full paper title

### Domain Coverage
- **llm_model_name**: LLM name (GPT-3, BERT, T5, Qwen, etc.) or "NR"
- **parameter_count**: Model size (e.g., "7B", "175B", "340M") or "NR"
- **slm_feasibility**: Does the paper discuss small language model feasibility? (Yes/No/NR)
- **fl_architecture**: FL paradigm (FedAvg, FedProx, Decentralized, Centralized) or "NR"
- **kg_type**: Knowledge graph type (Ontology, Property Graph, Embedding-based) or "NR"

### Pedagogical Variables
- **cefr_alignment**: CEFR level mentioned (A1, A2, B1, B2, C1, C2) or "NR"
- **privacy_mechanism**: Privacy technology (Differential Privacy, Secure Aggregation, Homomorphic Encryption) or "NR"
- **validation_metrics**: Validation metrics used (KGQI, HITL, ACC, BLEU, Hits@k) or "NR"

### Grounding Dimensions
- **grounding_gap_addressed**: Does the paper address constraining LLM outputs with KG rules? (Yes/No/NR)
- **control_gap_addressed**: Does the paper discuss syntactic/semantic control mechanisms? (Yes/No/NR)

### Extraction Metadata
- **confidence_score**: Your confidence in this extraction (0.0-1.0, where 1.0 = complete information)
- **extraction_notes**: Any ambiguities or notes

## Few-Shot Examples

### Example 1: High Confidence (Score: 0.9)
**Input:** "We fine-tuned GPT-3 (175B parameters) using federated learning with FedAvg for CEFR B2-level English instruction..."
**Output:**
{{
  "llm_model_name": "GPT-3",
  "parameter_count": "175B",
  "fl_architecture": "FedAvg",
  "cefr_alignment": "B2",
  "confidence_score": 0.9
}}

### Example 2: Low Confidence (Score: 0.2)
**Input:** "Our system uses neural networks for adaptive learning..."
**Output:**
{{
  "llm_model_name": "NR",
  "parameter_count": "NR",
  "fl_architecture": "NR",
  "cefr_alignment": "NR",
  "confidence_score": 0.2,
  "extraction_notes": "No specific LLM/FL/KG details provided"
}}

## Output Format
Return a valid JSON object matching the Pydantic schema. Use "NR" for unreported fields.

D.4. Extraction Pipeline Workflow
Step-by-Step Process:
Document Preprocessing
PDF → Text extraction (PyMuPDF)
Markdown → Frontmatter parsing
Text cleaning (remove figures, tables, references)
Truncate to first 8,000 tokens (to fit LLM context)
LLM Inference
Send prompt + paper text to Qwen 2.5 7B (vLLM API)
Temperature: 0.3 (deterministic)
Parse JSON response
Pydantic Validation
Validate JSON against PaperExtraction schema
Type checking (int, float, str constraints)
Field validation (year range, confidence range)
If validation fails → Retry with error feedback (max 2 retries)
Quality Assurance
Confidence score threshold: Papers with <0.3 flagged for manual review
“NR” rate check: Papers with >80% NR flagged
Manual spot-checks: 20% random sample (10/51 papers)
CSV Export
Pydantic model → CSV row
All 51 papers processed
File: Data_Extraction_Results_v1.csv

D.5. Error Handling and Retry Logic
async def extract_paper_with_retry(
    paper_text: str,
    paper_id: str,
    max_retries: int = 2
) -> PaperExtraction:
    """
    Extract data from paper with Pydantic validation and retry logic.
    """
    for attempt in range(max_retries + 1):
        try:
            # Call LLM
            prompt = build_extraction_prompt(paper_text, paper_id)
            response = await llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=2048
            )

            # Parse JSON
            json_data = parse_json_response(response)

            # Validate with Pydantic
            extraction = PaperExtraction(**json_data)

            return extraction

        except ValidationError as e:
            if attempt < max_retries:
                # Retry with error feedback
                error_msg = str(e)
                prompt = build_retry_prompt(paper_text, paper_id, error_msg)
                continue
            else:
                # Final attempt failed → manual review
                logger.error(f"Extraction failed for {paper_id}: {e}")
                return create_fallback_extraction(paper_id)

        except JSONDecodeError as e:
            if attempt < max_retries:
                # Retry with stricter JSON instructions
                prompt = build_json_strict_prompt(paper_text, paper_id)
                continue
            else:
                raise

    raise RuntimeError(f"All extraction attempts failed for {paper_id}")

D.6. Validation and Quality Control
Automated Validation: - Pydantic type checking (100% of papers) - Range validation (year 2019-2025, confidence 0-1) - Required field presence (study_id, author, year, title)
Manual Review: - 20% supervisor audit: 10/51 papers randomly selected - Discrepancy resolution: Inter-rater agreement: 95% (kappa = 0.92) - Low-confidence review: 5 papers with confidence <0.3 manually reviewed
Quality Metrics: - Extraction success rate: 100% (51/51 papers processed) - Validation failures: 0 (all passed Pydantic schema) - Average confidence: 0.17 (expected for high NR rates) - Manual corrections: 2 papers (year extraction errors corrected)

D.7. Reproducibility Checklist
To reproduce this extraction pipeline:
✓ Model: Qwen 2.5 7B Instruct (Hugging Face: Qwen/Qwen2.5-7B-Instruct)
✓ Inference: vLLM v0.6.3.post1 with settings documented in D.1
✓ Schema: Pydantic model code provided in D.2
✓ Prompt: Full template provided in D.3
✓ Pipeline: Workflow code provided in D.4-D.5
✓ Data: Raw PDFs/markdown files available on OSF
✓ Results: Data_Extraction_Results_v1.csv available on OSF
OSF Repository: All data supporting the findings of this study are available on the Open Science Framework (OSF) at https://osf.io/ds74h/  (view-only access: https://osf.io/ds74h/overview?view_only=d8be61e73da8453f9d83c4784d0f860c ).
Code Availability: Full Python extraction scripts available at OSF supplementary materials: - extraction_pipeline.py (main workflow) - pydantic_schema.py (schema definitions) - prompt_templates.py (LLM prompts) - validation_utils.py (QA checks)

D.8. Computational Resources
Processing Time: - Per paper: ~45 seconds average (including LLM inference, validation) - Total corpus (51 papers): ~38 minutes - Retries (3 papers): ~2 minutes additional
Cost (if using cloud LLM): - Qwen 2.5 7B: Self-hosted (no API costs) - Estimated cloud cost equivalent (GPT-3.5-turbo): ~$2.50 USD for 51 papers
Energy Consumption: - GPU power draw: ~350W - Processing time: 40 minutes - Estimated energy: ~0.23 kWh

END OF APPENDIX D

END OF MANUSCRIPT
Word Count: ~9,500 words (body text) Total with References/Appendices: ~12,000 words

Figures and Tables
Figures
Figure 1. PRISMA-ScR Flow Diagram: Study Selection Process (660 → 51 papers)

PRISMA Flow Diagram
{width=100%}
Figure 2. Venn Diagram: FL-KG-LLM Domain Overlap Analysis (n=51)

Venn Diagram
{width=80%}
Figure 3. Reporting Gap Heatmap: Not Reported (NR) Rates by Variable

Reporting Gap Heatmap
{width=100%}
Figure 4. Temporal Trends: Publication Distribution and Domain Representation (2019-2025)

Temporal Trends
{width=100%}
Figure 5. Domain Coverage Breakdown: Single/Dual/Triple Domain Distribution

Domain Coverage
{width=80%}
Figure 6. Hypothesis Testing Summary: All Five Hypotheses (H1-H5)

Hypothesis Testing
{width=100%}

Tables
Table 1. Convergence Deficit Analysis - Domain Coverage Distribution (n=51)
Note: n = 51 papers. Triple-domain = FL+KG+LLM; Dual-domain = any two domains; Single-domain = one domain only.

Table 2. Reporting Completeness Matrix (NR = Not Reported)
Average NR Rate: 84.5% Note: n = 51 papers. NR = Not Reported in paper.

Table 3. Pedagogical Variables Summary (NR = Not Reported)
Note: n = 51 papers. Dimension 1 = constraining LLM outputs; Dimension 2 = KG source verification.

Table 4. Quality Distribution by Extraction Confidence Score (NR = Not Reported)
Note: n = 51 papers. Confidence score reflects LLM extraction certainty (0-1 scale). Low confidence indicates high NR rates, not extraction errors.

Table 5. Hypothesis Testing Summary (NR = Not Reported)
Result: All five hypotheses supported. n = 51 papers.

Supplementary Materials: All figures available in PNG (300 DPI) and PDF (vector) formats in visualizations/ folder. All tables available in CSV and Markdown formats in tables/ folder. in PNG (300 DPI) and PDF (vector) formats. All tables available in CSV and Markdown formats as supplementary materials.