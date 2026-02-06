# Aprendizado Ativo ciente de incerteza para adaptação contínua em sistemas VANT-borda

Este repositório contém os scripts e instruções para reproduzir os experimentos do artigo:
**“Aprendizado ativo ciente de incerteza para adaptação contínua em sistemas VANT-borda”**.

A avaliação simula um ciclo de aprendizado em ciclo fechado com orçamento fixo de anotação, comparando:
- **Amostragem Aleatória** (`random`)
- **Amostragem por Incerteza (passagem única)** (`us_single`)
- **Amostragem por Incerteza Bayesiana (MC Dropout)**, método proposto (`proposed_mc`)

---

## Estrutura de pastas esperada

O código assume a seguinte estrutura na raiz do repositório:

- `data_raw/`
  - `data.yaml`
  - `train/`
    - `images/`
    - `labels/`
  - `test/`
    - `images/`
    - `labels/`
- `splits/` 
- `runs/` 
- `scripts/` 

Os scripts principais:
- `prepare_splits.py`: cria o split inicial L0, validação e pool não rotulado
- `run_cycles.py`: executa os ciclos de aprendizado ativo e salva logs por método

---

## Requisitos

Recomendação de ambiente:
- Python **3.10**
- GPU NVIDIA (opcional, mas recomendado para velocidade)
- CUDA compatível com sua instalação do PyTorch (se usar GPU)

Instale dependências:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
