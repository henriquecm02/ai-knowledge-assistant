# 📚 AI Knowledge Assistant: RAG sobre documentos internos, com fontes verificadas

**Um assistente que responde dúvidas de colaboradores a partir das políticas da empresa (reembolso, férias, segurança, onboarding, conduta, TI), sempre mostrando de qual documento, seção e página veio a resposta, e que diz "não encontrei" quando a informação não está lá.**

É a evolução do meu [chatbot-pdf](https://github.com/henriquecm02/chatbot-pdf), que criava o índice vetorial mas nunca o usava na resposta. Aqui a recuperação é medida antes de qualquer geração.

## O problema

As políticas internas ficam espalhadas em PDFs e páginas que ninguém lê inteiras. As mesmas perguntas ("qual o limite de hospedagem?", "quando peço férias?") chegam ao RH, ao financeiro e ao suporte toda semana. Um chatbot genérico **inventa respostas plausíveis**, o que é pior do que não responder.

**Usuário-alvo:** colaboradores (quem pergunta) e as áreas que mantêm as políticas (quem para de responder sempre a mesma coisa).

## Como funciona

```mermaid
flowchart LR
    D[PDF · DOCX · MD · TXT] --> C[Chunking por seção e página<br/>com sobreposição]
    C --> B[(BM25)]
    C --> E[(Embeddings Gemini<br/>matriz numpy)]
    Q[Pergunta] --> B & E
    B & E --> F[Fusão RRF]
    F --> R{Reordenar com IA?}
    R -->|nota ≥ 2| G[Gemini: resposta + citações<br/>saída JSON]
    R -->|não| G
    F -->|nada recuperado| N[Não encontrei<br/>sem chamar o LLM]
    G --> V[Validação em código<br/>citações existem? há fonte?]
    V --> A[Resposta + fontes]
```

| Etapa | Decisão | Por quê |
|---|---|---|
| Chunking | por seção (títulos Markdown ou numeração em PDF) e página, corte só quando a seção passa de 900 caracteres | cada trecho vira uma citação legível: `politica_reembolso.pdf · 2. Limites · p. 1` |
| Busca léxica | BM25 próprio, com remoção de acentos, stopwords e *stemming* leve para português | acerta valores, siglas e termos exatos; funciona sem modelo |
| Busca semântica | embeddings do Gemini (`RETRIEVAL_DOCUMENT` e `RETRIEVAL_QUERY`) em matriz numpy | acerta paráfrases. Até dezenas de milhares de trechos, a busca exata é rápida e dispensa banco vetorial |
| Fusão | Reciprocal Rank Fusion | combina rankings sem calibrar escalas de score diferentes |
| Reordenação | opcional: o LLM dá nota de 0 a 3 a cada trecho e os abaixo de 2 são descartados | aumenta a precisão quando há muitos trechos parecidos |
| Geração | JSON com `resposta`, `encontrou`, `citacoes` | estrutura verificável |
| Garantias em código | citações a trechos inexistentes são removidas; "encontrei" sem citação vira "não verificado"; sem trecho recuperado, o LLM não é chamado | o modelo não decide sozinho se a resposta é confiável |

## Avaliação

[`eval/perguntas.json`](eval/perguntas.json) traz 38 perguntas com o trecho correto marcado:

- **20 diretas**, que usam as palavras do documento;
- **14 paráfrases**, como "posso pegar Uber voltando tarde?" para a regra de táxi entre 22h e 6h;
- **4 sem resposta** na base, em que o correto é se abster.

### Recuperação (medida)

`python eval/run_eval.py --embeddings` ([resultado](eval/resultados/)); embeddings `gemini-embedding-001`, medido em 16/11/2026:

| Modo | Perguntas | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|
| BM25 | diretas (20) | 1,00 | 1,00 | 1,00 | 1,00 |
| BM25 | **paráfrases (14)** | **0,64** | 0,79 | 0,86 | 0,73 |
| Denso (Gemini) | diretas (20) | 1,00 | 1,00 | 1,00 | 1,00 |
| Denso (Gemini) | **paráfrases (14)** | **1,00** | 1,00 | 1,00 | 1,00 |
| Híbrido (RRF) | diretas (20) | 1,00 | 1,00 | 1,00 | 1,00 |
| Híbrido (RRF) | **paráfrases (14)** | **0,79** | 0,93 | 0,93 | 0,86 |

**Leitura honesta:** as perguntas diretas são fáceis por construção, porque eu escrevi a base e as perguntas. O número informativo é o das paráfrases: o BM25 erra o primeiro trecho em 5 de 14 e não encontra nada para "ajuda de custo para trabalhar de casa" (o documento diz "auxílio home office"). É exatamente a lacuna que a busca semântica deveria fechar, e fechou: o denso acerta as 14 paráfrases em primeiro lugar.

**Achado contra a expectativa:** nesta base, o **híbrido ficou pior que o denso** (R@1 de 0,79 contra 1,00 nas paráfrases). O BM25 puxa para o trecho errado em 3 perguntas (21, 22 e 26) e o RRF mistura esse ruído no ranking. Com 34 trechos e perguntas escritas por mim, a amostra é pequena demais para concluir que o denso seja sempre melhor, e não mudei o modo padrão do assistente com base nisso; avaliar com uma base maior está no roadmap. Não ajustei o BM25 olhando essas perguntas, para não inflar o resultado.

![Modo busca](docs/img/busca.png)

> Captura do modo busca (sem IA) para a paráfrase "meu notebook foi roubado no ônibus": o trecho correto ("Incidentes": comunicar em até 1 hora) aparece só em 4º. Esse é o caso que embeddings e reordenação devem resolver.

### Respostas (Gemini real, 38 perguntas)

`python eval/run_eval.py --llm --rerank --denso`, modelo `gemini-3.5-flash-lite`, 16/11/2026. Cada pergunta conta como correta se a resposta **contém o fato esperado e cita o trecho certo** (ou, nas sem resposta, se o assistente **se abstém**). Respostas completas em [`eval/resultados/respostas_gemini-3.5-flash-lite.json`](eval/resultados/).

| Tipo | Perguntas | Híbrido | Híbrido + reordenação | Denso |
|---|---|---|---|---|
| Diretas | 20 | 20/20 | 20/20 | 20/20 |
| Paráfrases | 14 | 13/14 | **14/14** | **14/14** |
| Sem resposta (deve se abster) | 4 | 4/4 | 4/4 | 4/4 |
| **Total** | **38** | **37/38** | **38/38** | **38/38** |

| Custo medido | Híbrido | Híbrido + reordenação | Denso |
|---|---|---|---|
| Chamadas ao modelo (38 perguntas) | 38 | 72 (média de 1,9 por pergunta) | 38 |
| Tempo por resposta, mediana | 1,4 s | 4,0 s | 1,2 s |
| Tempo por resposta, p90 | 8,9 s | 23,7 s | 23,7 s |

O p90 do denso é igual ao do reranker por causa de picos de latência da API em poucas perguntas, não por trabalho extra (o denso faz 1 chamada por pergunta); com 38 amostras, o p90 é só 4 perguntas lentas e não diz nada sobre o modo.

**O único erro do híbrido** é a pergunta 22 ("posso pegar Uber voltando tarde da noite de um evento?"). É falha de **recuperação**, não de geração: o BM25 puxou o trecho errado ("Redes sociais") para o primeiro lugar, o assistente não achou o limite de 22h e respondeu que a informação não está nos documentos. Falhou do jeito seguro, sem inventar. A reordenação por LLM e a busca densa corrigiram esse caso, e foi o único que mudou entre os modos.

**Nas 4 perguntas sem resposta**, os três modos se abstiveram. Há uma diferença de mecanismo: com a reordenação, o assistente chegou a recusar sem gerar resposta ("nenhum trecho recuperado"), porque o reranker descarta trechos irrelevantes. No modo denso a busca **sempre devolve trechos** e a abstenção depende inteiramente do modelo, que recusou as 4 (nenhuma resposta inventada). Isso vale para este modelo e estas perguntas; não é uma garantia do modo.

**Leitura honesta do resultado:**

- Nesta base, o **denso iguala o melhor resultado (38/38) pelo custo do híbrido** (38 chamadas, sem a chamada extra de reordenação). É plausível que seja o melhor modo padrão aqui, e a recuperação medida aponta na mesma direção (R@1 de 1,00 contra 0,79 do híbrido nas paráfrases). **Não mudei o modo padrão do assistente** por causa disso.
- O teste está **saturado**: três modos entre 37 e 38 acertos em 38 não se distinguem, uma pergunta é a diferença toda, e o híbrido e o reranker foram medidos antes do denso (saída não determinística). Separar os modos de verdade exige uma base maior e perguntas mais difíceis.
- 38 perguntas escritas por mim sobre uma base fictícia de 34 trechos, um modelo, uma rodada. Mostra o fluxo completo funcionando e onde ele falha; não é uma medida de acurácia em produção.

## Status da validação

| Componente | Status |
|---|---|
| Leitura de PDF, DOCX e MD, chunking com página e seção | ✅ testado |
| BM25 e avaliação de recuperação | ✅ medido (tabela acima) |
| Garantias de citação e abstenção | ✅ testado com LLM roteirizado (11 testes) |
| Modo busca na interface (sem chave) | ✅ executado |
| Embeddings, busca densa e híbrida com o **Gemini real** | ✅ medido nas 38 perguntas |
| Respostas do Gemini (híbrido, com reordenação e denso) | ✅ 38 perguntas: híbrido 37/38, com reordenação 38/38, denso 38/38; veja acima |
| Comparação entre modelos | ⏳ **pendente** |

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ui,dev]"
pytest -q
python eval/run_eval.py                 # avaliação da recuperação (sem chave)
streamlit run app/streamlit_app.py      # modo busca sem chave; com GEMINI_API_KEY, respostas geradas
```

Para usar com os seus documentos, envie os arquivos pela barra lateral da interface ou coloque-os em `corpus/`.

## Limitações

- PDFs escaneados (imagem) não têm texto extraível; precisaria de OCR antes.
- Tabelas complexas em PDF perdem a estrutura na extração de texto.
- O índice é reconstruído a cada inicialização (rápido para dezenas de documentos; para milhares, persistir os vetores).
- Base de demonstração pequena (6 documentos, 34 trechos), de uma empresa fictícia.

## Roadmap

- [ ] Comparar modelos
- [ ] Reavaliar denso × híbrido × reordenação numa base maior e mais difícil (a atual está saturada) e então decidir o modo padrão; hoje o denso empata com o melhor pelo menor custo
- [ ] Comparação entre documentos ("o que mudou da versão 3.1 para a 3.2?")
- [ ] Persistir índice e embeddings (SQLite) para bases maiores
- [ ] Feedback 👍/👎 na interface para alimentar o conjunto de avaliação

---

Todos os documentos de `corpus/` são de uma empresa **fictícia**. Licença MIT.
