# Quant Brain

Servico Python isolado para analise quantitativa, telemetria estrategica e IA tatico/estrategica do ecossistema BingX.

Este projeto roda separado da raiz e separado do backend Node.js. A raiz do monorepo nao deve commitar `quant-brain/`.

## Rodar Local

```bash
cd quant-brain
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Healthcheck:

```text
GET http://localhost:9000/health
```

## Variaveis

| Variavel | Padrao | Uso |
|---|---:|---|
| `PORT` | `9000` | Porta HTTP do FastAPI/Uvicorn. |
| `ANTHROPIC_API_KEY` | vazio | Habilita analises com IA quando configurada. |

## Dados Locais

O arquivo `data/knowledge.db` e runtime data local. Ele guarda conhecimento operacional, padroes e estatisticas. Nao deve ir para Git.

## Endpoints

```text
GET  /health
GET  /market/snapshots
GET  /market/snapshots/{symbol}
GET  /market/anomalies
GET  /tactical/alerts
POST /tactical/analyze
GET  /strategic/report
GET  /strategic/edge-evolution
POST /strategic/analyze
POST /strategic/hypotheses
GET  /kb/patterns
GET  /kb/observations
GET  /kb/insights
GET  /kb/stats
GET  /kb/stats/{symbol}
POST /kb/trades
GET  /kb/feature-history/{symbol}
```

## Git

Este diretorio deve ser um repositorio Git proprio:

```bash
cd quant-brain
git add .
git commit -m "feat: update quant brain"
git push
```
