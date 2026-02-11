# PINNs for PDEs  
### Physics-Informed Neural Networks for Poisson and Fisher–KPP Equations

This repository contains a modular implementation of learning-based methods
for solving partial differential equations (PDEs), including:

- Physics-Informed Neural Networks (PINNs)
- Linear PINNs (basis expansion / variational approach)
- SGD-based variational solvers

The project focuses on the **Poisson equation** and the **stationary Fisher–KPP reaction–diffusion equation**.

⚠️ Classical numerical methods (e.g. finite differences) are **not implemented**
in this repository. They are only used externally to generate reference datasets.

---

# Project Goals

- Compare different learning-based PDE solvers
- Separate clearly:
  - PDE definition
  - solution parameterization
  - loss formulation
  - optimization strategy
- Provide a modular and extensible research framework

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/pinn-pde.git
cd pinn-pde

---

# Structure

EA_Physics-Informed_Learning_for_Reaction_Diffusion_Equations/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── equations.py
│   ├── generate_data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── models.py
│   ├── operators.py
│   ├── sampling.py
│   └── solvers.py
│
├── scripts/
│   └── run_experiment.py
│
├── notebooks/
│   └── visualization.ipynb
│
└── tests/
    └── test_operators.py
