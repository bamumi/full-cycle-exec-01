# Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema de busca em PDFs via linha de comando usando RAG (Retrieval-Augmented Generation) com LangChain, PostgreSQL + pgVector e Google Gemini.

## Pré-requisitos

- Python 3.10+
- Docker & Docker Compose
- Chave de API do Google Gemini

## Configuração

### 1. Criar e ativar o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

> No Windows (PowerShell): `venv\Scripts\Activate.ps1`

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` e preencha `GOOGLE_API_KEY` com sua chave do Gemini. Os demais valores já estão configurados para o Docker local.

## Ordem de execução

### 1. Subir o banco de dados

```bash
docker compose up -d
```

Aguarde o container `bootstrap_vector_ext` finalizar — ele habilita a extensão `pgvector` automaticamente.

### 2. Executar ingestão do PDF

```bash
python src/ingest.py
```

O script carrega `document.pdf`, divide em chunks de 1000 caracteres (overlap 150), gera embeddings via Gemini e armazena no PostgreSQL com pgVector.

### 3. Rodar o chat

```bash
python src/chat.py
```

## Exemplo de uso

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

PERGUNTA: sair
```

Digite `sair`, `exit` ou `quit` para encerrar, ou use `Ctrl+C`.
