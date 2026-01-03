---
# index.md
# Purpose: Main landing page for Federated Learning & Knowledge Graph research portfolio.
# Date/Time: Thursday, December 25, 2025, 12:00 PM Athens Time

layout: home
title: My Research Portfolio
---

# Federated Learning & Knowledge Graphs Research

## Affiliation
**Intelligent Systems Laboratory (i-lab)**
**Semantic Web of Things (SWoT) Group**
Department of Information & Communication Systems Engineering
University of the Aegean

https://i-lab.aegean.gr/swot/

---

## Description
This repository contains the codebase for post-doc research focusing on the intersection of **Federated Learning (FL)** and **Knowledge Graphs (KG)**.

---

## LICENSE & NOTICE

⚠️ **Pre-publication Code** - This repository contains unpublished research.

**Copyright © 2025 University of the Aegean. All Rights Reserved.**

Do NOT use, copy, modify, distribute, or cite without explicit permission from the author.

**Academic Integrity Notice:**
This work is under review for publication. Using this research without attribution or prior consent constitutes academic misconduct.

**For permissions, contact:** mkenteris@aegean.gr

---

## Tech Stack
* **Python**: Core logic for FL algorithms.
* **Neo4j**: Graph database for KG representation.
* **Google API**: Cloud integration services.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Configure `.env` with your API keys.

---

## Recent Articles
<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> — {{ post.date | date: "%B %d, %Y" }}
    </li>
  {% endfor %}
</ul>
