# CausaFormer

CausaFormer is an extension of the Transformer architecture designed to understand **causal relationships** in sequential data.  
It introduces three new components:
1. **Causal Graph Encoder (CGE)**: Learns latent causal structures between input tokens.
2. **Interventional Attention (IA)**: Simulates interventions to evaluate causal effects.
3. **Causal Compositionality Layer (CC)**: Integrates multiple causal relationships.

This repository provides a prototype implementation of the CausaFormer model in PyTorch.

---

## 🚀 Features
- Extension of standard Transformer with causal reasoning capability.
- Implements **Interventional Attention** to simulate interventions on token inputs.
- Flexible and modular design for research and experimentation.

---

## 📦 Installation

To get started, clone this repository and install the required libraries:

```bash
git clone https://github.com/yourusername/CausaFormer.git
cd CausaFormer
pip install -r requirements.txt
