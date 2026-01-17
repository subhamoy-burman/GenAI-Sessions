# GenAI Deep Dive: Introduction Session (Class 1 of 16)

## Overview
Document the foundational concepts from the introduction class. This is the first of a 16-part series, focusing only on establishing the mental model and core primitives.

---

## Single Document Structure

**File:** `session-01-introduction.md`

This consolidated document will follow a linear lecture flow, covering all introduction topics in a single, cohesive narrative.

---

## Document Outline

### **Section 1: The Ecosystem - Research vs. Application Layer**
- Define the two engineering disciplines
- Your role as an Application Engineer
- Engineering analogy: Combustion Engine (LLM) vs. Car (Product)
- **Practical Example Walkthrough:** Comparing AWS RDS (managed service) vs. building your own database engine - parallel to using OpenAI API vs. training your own LLM
- **Checkpoint Question:** Build vs. Buy decision (Uber scenario)

---

### **Section 2: The Core Primitive - Next Token Prediction**
- The stateless function: `f(context) → next_token`
- Probability distribution over vocabulary
- Backend parallel: REST APIs (stateless)
- **Practical Example Walkthrough:** Step-by-step trace of "The cat sat on the" → predicting "mat" with probability scores (cat: 5%, mat: 65%, floor: 20%, ...)
- **Checkpoint Question:** WhatsApp conversation history architecture

---

### **Section 3: Deconstructing GPT**
- **G**enerative: Creates vs. retrieves data
- **P**re-trained: The frozen binary concept (knowledge cutoff)
- **T**ransformer: The architecture name
- **Practical Example Walkthrough:** Ask GPT-3.5 (trained on data until Sept 2021) about Python 2.7 (correct) vs. asking about a library released in 2023 (hallucination) - demonstrating the knowledge cutoff
- **Checkpoint Question:** Why nightly retraining fails (Bloomberg news)

---

### **Section 4: The Architecture Pipeline (The "Black Box" Revealed)**
- **Tokenization:** Text → Integers (Serialization analogy)
- **Vector Embeddings:** Semantic space representation
- **Positional Encoding:** Sequence awareness
- **Attention Mechanism:** Self-attention & Multi-head
- **Inference Control:** Softmax & Temperature
- **Practical Example Walkthrough:** Complete trace of "The animal didn't cross the street because it was too tired"
  - Tokenization: ["The", "animal", "didn't", "cross", ...]
  - Embeddings: Show vector representations
  - Attention: How "it" links to "animal" (not "street")
  - Output: Next token prediction with temperature variations
- **Mermaid Diagram:** Complete pipeline flow (Input → Output)
- **Checkpoint Questions:** 
  - Amazon semantic search (Vector DB requirement)
  - Code review bot temperature setting

---

### **Section 5: Context Limitation & RAG**
- The context window problem
- RAG architecture: Retrieval → Augmentation → Generation
- External knowledge injection pattern
- **Practical Example Walkthrough:** Internal documentation chatbot scenario
  - User asks: "What's our vacation policy?"
  - Step 1 (Retrieval): Query vector DB for relevant HR docs
  - Step 2 (Augmentation): Inject retrieved context into prompt
  - Step 3 (Generation): LLM answers using only provided context
  - Show actual prompt structure with injected context
- **Mermaid Diagram:** RAG flow diagram
- **Checkpoint Question:** Google HR bot security (ACL enforcement)

---

### **Section 6: Final Architecture Review - Role Perspectives**
Brief insights for different engineering roles:
- **Backend/Cloud:** Streaming responses (SSE/WebSockets)
  - **Practical Example:** Show code snippet of SSE implementation
- **DevOps:** Quantization basics (VRAM management)
  - **Practical Example:** Llama-3-70B: 140GB (16-bit) → 40GB (4-bit) calculation
- **SRE:** Golden Signals (TTFT, Token Throughput)
  - **Practical Example:** Sample Grafana dashboard metrics
- **Platform:** AI Gateway concept
  - **Practical Example:** Request flow through gateway with rate limiting
- **Leadership:** Build vs. Buy framework
  - **Practical Example:** Cost comparison table (training vs. API)

---

### **Section 7: Practical Exercise - Build a Tokenizer**
**Single hands-on exercise to solidify understanding:**

- **Exercise:** Build a simple tokenizer from scratch
- **Python starter code template** provided
- **Mermaid Sequence Diagram:** Tokenization flow
- **Why this matters:** Connection to real-world LLM engineering
- **Expected output & validation**

---

## Document Structure Template

```markdown
# Session 1: Engineering Generative AI - Introduction

## Table of Contents
[Auto-generated from sections]

## 1. The Ecosystem: Research vs. Application Layer
[Content with analogy]
### Practical Example Walkthrough
[Real-world example demonstrating the concept]
### Checkpoint Question
[Scenario with detailed answer]

## 2. The Core Primitive: Next Token Prediction
[Content with backend parallels]
### Practical Example Walkthrough
[Step-by-step trace with probability scores]
### Checkpoint Question
[Scenario with detailed answer]

## 3. Deconstructing GPT
[Content]
### Practical Example Walkthrough
[Knowledge cutoff demonstration]
### Checkpoint Question
[Scenario with detailed answer]

## 4. The Architecture Pipeline
[All sub-components covered]
### Practical Example Walkthrough
[Complete sentence trace through entire pipeline]
### Pipeline Flow Diagram
[Mermaid diagram]
### Checkpoint Questions
[2 scenarios with answers]

## 5. Context Limitation & RAG
[Content]
### Practical Example Walkthrough
[HR chatbot example with actual prompt structure]
### RAG Architecture Diagram
[Mermaid diagram]
### Checkpoint Question
[Scenario with answer]

## 6. Role Perspectives
[5 role-specific insights with mini practical examples each]

## 7. Hands-On Exercise: Build a Tokenizer
[Complete exercise with code, diagram, explanation]

## Summary & Key Takeaways
[Bullet points of core concepts]

## What's Next (Class 2 Preview)
[Brief teaser for next session]
```

---

## Revised Scope (Quality-First Approach)

- **Total Documents:** 1 consolidated lecture document
- **Total Word Count:** ~15,000-20,000 words (comprehensive, deep-dive)
- **Diagrams:** 4-5 Mermaid diagrams (Pipeline, RAG, Tokenizer flow, Attention visualization, Gateway architecture)
- **Code Examples:** 1 complete exercise at the end + inline snippets throughout practical walkthroughs
- **Checkpoint Questions:** 7 scenarios with detailed, production-grade answers
- **Philosophy:** Depth over brevity - every concept explained from first principles with real-world engineering context

---

## Success Criteria (Introduction Class)

After completing this session document, students should be able to:

1. ✅ Articulate the difference between Research and Application layer
2. ✅ Explain LLMs as stateless functions
3. ✅ Understand what "Pre-trained" means architecturally
4. ✅ Describe the token → embedding → attention → output pipeline
5. ✅ Explain when and why RAG is needed
6. ✅ Make basic Build vs. Buy arguments
7. ✅ Implement a basic tokenizer from scratch

---

## Delivery Format

This single document is designed for:
- **Extended Lecture delivery** (2+ hour deep-dive session, can be broken into multiple parts)
- **Self-study** (Principal Engineers can read through at their own pace)
- **Workshop format** (with live coding of the final exercise)

The checkpoint questions can be:
- Discussed during lecture as interactive problem-solving sessions
- Used as pause points for Q&A and deeper exploration
- Given as homework/reflection prompts for asynchronous learning

**Quality Focus:**
- No rushing through concepts - each section gets the time it deserves
- Detailed explanations with backend/distributed systems parallels
- Real production scenarios and architectural trade-offs
- Deep technical accuracy over superficial coverage

---

## Next Steps

1. **Review this plan** - Approve or request modifications
2. **Create session-01-introduction.md** - The complete consolidated document
3. **Review the output** - Ensure lecture flow is correct
4. **Prepare for Class 2** - Use this as foundation

---

*Session 1: Establishing the mental model. Linear narrative. One practical exercise. Ready for classroom delivery.*
