---
# index.md
# Purpose: Main landing page for Federated Learning & Knowledge Graph research portfolio.
# Date/Time: Thursday, December 25, 2025, 12:00 PM Athens Time

layout: home
title: My Research Portfolio
---

# Federated Learning & Knowledge Graphs Research

## Description
This repository contains the codebase for post-doc research focusing on the intersection of **Federated Learning (FL)** and **Knowledge Graphs (KG)**. 

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
