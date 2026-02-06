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
- `plots/` (Após execução do script plot_cycles.py)
- `tables/` (Após execução do script compile_tables.py)
- `runs/` (Após execução do script run_cycles.py)
- `splits/` (Após execução do script prepare_splits.py)
- `scripts/` 

Os scripts principais:
- `prepare_splits.py`: cria o split inicial L0, validação e pool não rotulado
- `run_cycles.py`: executa os ciclos de aprendizado ativo e salva logs por método
- `run_configurations.py`: executa os dois scripts acima de acordo com a configuração escolhida (L0, budget, cycles).
- `plot_cycles.py`: cria os gráficos de evolução dos ciclos, utilize a seguinte linha de comando na raiz:
python scripts/plot_cycles.py   --exp_dir runs/nome_da_pasta_da_run   --metrica mAP50   --x cycle   --out_dir plots
- `compile_tables.py`: cria as tabelas que compilam os resultados obtidos das métricas.
---

## Requisitos

Recomendação de ambiente:
- Python **3.10**
- GPU NVIDIA 
- CUDA compatível com sua instalação do PyTorch (se usar GPU)

Instale dependências:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

## Copyright

Copyright 2026 - ATLab – Laboratório Alan Turing
Universidade Federal do Ceará (UFC)
https://www.atlab.ufc.br

This software is licensed under the terms of the GNU Affero General Public License, either version 3 of the License, or (at your option) any later version, for non-commercial purposes.

Este software é licenciado sob os termos da licença 
GNU Affero General Public License, tanto na versão 3 da Licença, ou (por sua escolha) qualquer outra versão mais recente, para fins não comerciais.