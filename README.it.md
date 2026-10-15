# rizzo-flow-triage

[English](README.md) · Italiano

[rizzo-flow](https://github.com/Rizzo-AI-Academy/rizzo-flow) su una **NVIDIA
Jetson Orin Nano Super**: le modifiche servite per farlo girare sulla scheda, i
test effettuati e i risultati.

rizzo-flow trasforma un LLM in una fonte di decisioni tipizzate: ogni risposta
ammessa diventa una lettera e la decisione si legge dalle probabilita' di quelle
lettere, senza generare token. Il suo README chiede report da Linux e da
hardware nuovo; questo repository e' uno di questi, per una Jetson Orin Nano
Super (8 GB condivisi fra CPU e GPU, modalita' 25 W, JetPack 6).

## Cosa e' stato modificato

La scheda ha JetPack 6 (L4T R36.4.7, CUDA 12.6). Tre ostacoli, nell'ordine in
cui sono comparsi.

1. **Il runtime precompilato richiede una CUDA piu' nuova.** Su Linux arm64
   con GPU NVIDIA, `rizzo download` sceglie la build ufficiale di llama.cpp
   `ubuntu-cuda-13.4-arm64`, che richiede un driver CUDA 13; JetPack 6 ha CUDA
   12.6. Quella build non l'ho provata: ho compilato llama.cpp dai sorgenti al
   commit che rizzo-flow fissa (`161755f`), con CUDA per la GPU della Orin
   (`sm_87`) e solo la libreria condivisa `llama`, e l'ho indicata a rizzo-flow
   con `RIZZO_LLAMA_DIR`, come prevede il suo README. La compilazione sulla
   scheda ha richiesto 34 minuti.
2. **Il modello era in memoria due volte.** Sulla Jetson CPU e GPU usano la
   stessa RAM. Con il caricamento predefinito llama.cpp mappa il file GGUF
   (2,6 GB in page cache) e copia i pesi sulla GPU: qui le due copie stanno
   negli stessi 8 GB, e l'allocazione della KV cache falliva
   (`NvMapMemAllocInternalTagged ... error 12`, `cudaMalloc failed: out of
   memory`) anche con 2 GB liberi e contesto ridotto. Togliere il file dalla
   page cache con `posix_fadvise` non bastava: il caricamento la riempie di
   nuovo.
3. **Caricare senza mmap: la patch.** llama.cpp a quel commit puo' leggere il
   modello con I/O diretto (`LLAMA_LOAD_MODE_DIRECT_IO = 4` in `llama.h`), che
   salta la page cache, ma rizzo-flow non espone l'impostazione.
   `patch/rizzo-flow-load-mode.patch` aggiunge quattro righe a
   `src/rizzo_flow/llama_cpp.py`: se la variabile d'ambiente
   `RIZZO_LLAMA_LOAD_MODE` e' impostata, il suo valore va in
   `model_params.load_mode` prima di caricare il modello. Senza variabile non
   cambia nulla. Con `RIZZO_LLAMA_LOAD_MODE=4` e `--ctx 1024` il server parte,
   con un picco di 2,68 GB sul dispositivo.

```diff
         model_params.n_gpu_layers = GPU_LAYERS if chosen else 0
+        # Jetson and other integrated GPUs: with mmap the GGUF takes RAM twice (page
+        # cache + GPU copy) and cudaMalloc fails. RIZZO_LLAMA_LOAD_MODE=4 uses direct I/O.
+        if os.environ.get("RIZZO_LLAMA_LOAD_MODE"):
+            model_params.load_mode = int(os.environ["RIZZO_LLAMA_LOAD_MODE"])
         model = library.llama_model_load_from_file(str(gguf).encode(), model_params)
```

## Test

Il carico di prova e' il triage di richieste interne in italiano: tre decisioni
tipizzate per ogni richiesta.

| Decisione | Tipo | Valori |
| --- | --- | --- |
| `reparto` | choice | `salesforce`, `m365`, `report`, `it` |
| `tipo` | choice | `guasto`, `richiesta`, `domanda` |
| `urgente` | boolean | blocca qualcuno con scadenza entro oggi/domani o ferma piu' persone |

Le tre domande condividono lo stesso stato (il testo della richiesta): il
prefisso si calcola una volta, poi rizzo-flow legge solo i logit delle lettere
di risposta ammesse. Per confronto le stesse richieste passano da Qwen 3.5 1,9B
Q8_0 su Ollama, sulla stessa scheda, che genera un JSON vincolato da uno
schema. I due motori vedono esattamente le stesse descrizioni di reparti e
tipi, definite in `triage.py`.

`dataset/richieste.jsonl`: 40 richieste **sintetiche**, scritte per questa
prova, 10 per reparto; 19 guasti, 12 richieste, 9 domande; 12 urgenti. Le
etichette le ha scritte l'autore del dataset, non un gruppo di annotatori: e'
una dimostrazione, non un benchmark.

## Risultati — 2026-10-12, Jetson Orin Nano Super, modalita' 25 W

| | rizzo-flow 4B Q4_K_M | Qwen 3.5 1,9B Q8_0 su Ollama |
| --- | ---: | ---: |
| Come decide | logit delle risposte ammesse | JSON generato con schema |
| `reparto` | **40/40** | **40/40** |
| `tipo` | **35/40** (87,5%) | 31/40 (77,5%) |
| `urgente` | 35/40 (87,5%) | **39/40** (97,5%) |
| tutte e tre giuste | 31/40 (77,5%) | 30/40 (75%) |
| ms per richiesta, p50 / p95 | **1044 / 1068** | 1457 / 1587 |
| decisioni al secondo | **2,86** | 2,05 |
| token generati | **0** | 1180 |
| probabilita' per ogni opzione | si' | no |

Dettaglio rizzo-flow (mediane per richiesta): prefill dello stato 0,3 s,
inferenza totale 1,0 s, 144 token di prefisso condiviso fra le tre domande.

Lettura onesta:

- **Il reparto e' facile per entrambi**: le descrizioni nominano i sistemi.
- **rizzo-flow sbaglia per prudenza sull'urgenza** (4 urgenti non viste: R14,
  R21, R26, R33; un falso allarme, R07) e tende a "richiesta" sul tipo. Qwen
  fa il contrario: urgenza quasi perfetta, tipo piu' confuso (8 errori su 9
  coinvolgono "domanda", in entrambe le direzioni).
- **Con 40 casi una richiesta vale 2,5 punti**: le differenze di accuratezza
  non sono significative. La differenza di latenza (circa 30%) e i token a
  zero sono invece sistematici.
- **Modelli diversi**: 4B contro 1,9B. Il confronto e' fra due modi di usare
  un LLM su questo hardware, non fra due modelli a parita' di taglia.
- Le probabilita' di rizzo-flow non sono calibrate (README del progetto): per
  usarle come soglie servono dati etichettati del dominio.
- Sulla GPU del README (RTX 5060 Ti) rizzo-flow dichiara circa 50 ms per una
  decisione breve; qui sono circa 350 ms per decisione: la Orin Nano e' una
  scheda da 25 W.

File: `risultati/rizzo-jetson-q4_k_m.json`, `risultati/ollama-qwen3.5-1.9b-q8_0.json`
(ogni riga con atteso, predetto, tempi; rizzo-flow anche probabilita' e
`timing`).

## Installazione sulla Jetson

```bash
mkdir -p ~/rizzo-flow-lab && cd ~/rizzo-flow-lab
git clone https://github.com/rimentica/rizzo-flow-triage.git triage
git clone https://github.com/Rizzo-AI-Academy/rizzo-flow.git
git -C rizzo-flow checkout b9ba007ee4d2928bbab5b1d8bfe9009c3696b6de
git -C rizzo-flow apply ../triage/patch/rizzo-flow-load-mode.patch
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git
git -C llama.cpp checkout 161755f29e415e2c33efe906e91843c068efd664
python3 -m pip install --user uv

# solo libllama e ggml, condivise, CUDA per Orin (sm_87)
cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama -j 4

# ambiente Python e pesi (4B fine-tuned, Q4_K_M 2,6 GB)
cd ../rizzo-flow
~/.local/bin/uv sync --locked
~/.local/bin/uv run rizzo download --only weights --quant q4_k_m
```

Tutto sta in `~/rizzo-flow-lab`; fuori da quella cartella non cambia nulla.

## Esecuzione

`triage.py` usa solo la libreria standard di Python.

```bash
cd ~/rizzo-flow-lab/rizzo-flow
RIZZO_LLAMA_DIR=~/rizzo-flow-lab/llama.cpp/build/bin RIZZO_LLAMA_LOAD_MODE=4 \
  ~/.local/bin/uv run rizzo serve --quant q4_k_m --ctx 1024 --device cuda

# da un altro terminale
cd ~/rizzo-flow-lab/triage
python3 triage.py --rizzo http://127.0.0.1:8017 --out risultati/rizzo-jetson-q4_k_m.json
# fermare rizzo-flow, poi, con il nome del modello nel proprio Ollama
python3 triage.py --ollama MODELLO --out risultati/ollama.json
```

Su 8 GB condivisi fra CPU e GPU i due modelli non stanno caricati insieme: si
misura un motore alla volta.

## Possibili passi successivi

- Portare il load mode in rizzo-flow come opzione `--load-mode` invece che
  come patch: proposto in [rizzo-flow#31](https://github.com/Rizzo-AI-Academy/rizzo-flow/issues/31), in attesa del maintainer.
- Rendere piu' affidabile l'urgenza con istruzioni piu' precise, oppure
  calibrare le probabilita' su richieste etichettate.
- Provare il modello rizzo-flow piu' piccolo per la latenza.

## Licenza

Codice, dataset e risultati: Apache-2.0 (`LICENSE`). La patch in `patch/`
modifica un file di [rizzo-flow](https://github.com/Rizzo-AI-Academy/rizzo-flow),
anch'esso Apache-2.0; il file modificato porta un commento che descrive la
modifica. I pesi del modello non sono inclusi: si scaricano con `rizzo download`.
