#!/usr/bin/env python3
"""Triage di richieste interne: reparto, tipo e urgenza decisi da un LLM.

Due motori sullo stesso dataset e con le stesse etichette:
  - rizzo-flow: legge i logit delle risposte ammesse, zero token generati;
  - Ollama: il modello genera un JSON vincolato da uno schema (confronto).

Uso, sulla macchina che serve il modello (solo libreria standard):
    python3 triage.py --rizzo http://127.0.0.1:8017 --out risultati/rizzo.json
    python3 triage.py --ollama MODELLO --out risultati/ollama.json
"""
import argparse
import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Le descrizioni sono l'unica conoscenza del dominio che il modello riceve:
# entrambi i motori vedono esattamente gli stessi testi.
REPARTI = {
    "salesforce": "CRM Salesforce: account, opportunità, campagne, attività, permessi e profili Salesforce, add-in Outlook per Salesforce, Flow.",
    "m365": "Microsoft 365: SharePoint, Teams, Outlook e caselle condivise, OneDrive, condivisioni e accessi ai siti.",
    "report": "Report e dati: dashboard Power BI, SAP BW, SAP Analytics Cloud, estrazioni e report Excel automatici.",
    "it": "IT di base: PC, monitor, stampanti, rete e Wi-Fi, VPN, telefoni, account Windows.",
}
TIPI = {
    "guasto": "Qualcosa che prima funzionava ora non funziona o dà errore.",
    "richiesta": "Si chiede qualcosa di nuovo: un accesso, una modifica, un campo, un report, un'installazione.",
    "domanda": "Si chiede come fare qualcosa o una spiegazione, senza guasti e senza modifiche.",
}
DOMANDA_REPARTO = "Quale reparto deve gestire questa richiesta interna?"
DOMANDA_TIPO = "Di che tipo di richiesta si tratta?"
DOMANDA_URGENZA = ("La richiesta blocca il lavoro di qualcuno con una scadenza entro oggi "
                   "o domani, oppure ferma più persone?")
CAMPI = ("reparto", "tipo", "urgente")


def post(url, body, timeout=300):
    """POST JSON e risposta JSON, con il tempo di andata e ritorno in ms."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------- rizzo-flow
def rizzo_body(testo):
    """Tre domande sullo stesso stato: il prefisso si calcola una volta sola.
    Niente astensione: ogni richiesta del dataset ha una risposta."""
    policy = {"allow_abstain": False}
    opzioni = lambda d: [{"id": k, "description": v} for k, v in d.items()]
    return {"state": testo, "questions": {
        "reparto": {"type": "choice", "instructions": DOMANDA_REPARTO,
                    "options": opzioni(REPARTI), "policy": policy},
        "tipo": {"type": "choice", "instructions": DOMANDA_TIPO,
                 "options": opzioni(TIPI), "policy": policy},
        "urgente": {"type": "boolean", "instructions": DOMANDA_URGENZA, "policy": policy},
    }}


def rizzo(base, testo):
    out, ms = post(base.rstrip("/") + "/v1/decisions", rizzo_body(testo))
    a = out["answers"]
    pred = {"reparto": a["reparto"]["choice"], "tipo": a["tipo"]["choice"],
            "urgente": a["urgente"]["value"]}
    conf = {k: a[k]["uncertainty"]["top_probability"] for k in CAMPI}
    return pred, {"ms": ms, "confidenza": conf, "timing": out.get("timing"),
                  "token_generati": 0, "modello": out.get("model")}


# -------------------------------------------------------------------- Ollama
def ollama_body(model, testo):
    elenco = lambda d: "\n".join(f"- {k}: {v}" for k, v in d.items())
    system = (f"Classifica la richiesta interna dell'utente.\n{DOMANDA_REPARTO}\n{elenco(REPARTI)}\n"
              f"{DOMANDA_TIPO}\n{elenco(TIPI)}\nurgente: {DOMANDA_URGENZA}\n"
              "Rispondi solo con il JSON richiesto.")
    schema = {"type": "object", "required": list(CAMPI), "properties": {
        "reparto": {"type": "string", "enum": list(REPARTI)},
        "tipo": {"type": "string", "enum": list(TIPI)},
        "urgente": {"type": "boolean"}}}
    return {"model": model, "stream": False, "think": False, "format": schema,
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": testo}]}


def ollama(base, model, testo):
    out, ms = post(base.rstrip("/") + "/api/chat", ollama_body(model, testo))
    pred = json.loads(out["message"]["content"])
    return {k: pred.get(k) for k in CAMPI}, {
        "ms": ms, "token_generati": out.get("eval_count"),
        "modello": out.get("model")}


# ----------------------------------------------------------------- metriche
def percentile(valori, p):
    """Rango piu' vicino: con 40 campioni il p95 e' il 38esimo valore."""
    v = sorted(valori)
    return v[max(0, round(p / 100 * len(v)) - 1)]


def riepilogo(righe):
    n = len(righe)
    giusto = lambda r, k: r["predetto"][k] == r["atteso"][k]
    ms = [r["ms"] for r in righe]
    s = {"richieste": n, "decisioni": 3 * n}
    for k in CAMPI:
        s[f"accuratezza_{k}"] = round(sum(giusto(r, k) for r in righe) / n, 3)
    s["tutte_e_tre_giuste"] = round(sum(all(giusto(r, k) for k in CAMPI) for r in righe) / n, 3)
    s["ms_per_richiesta_p50"] = round(statistics.median(ms), 1)
    s["ms_per_richiesta_p95"] = round(percentile(ms, 95), 1)
    s["decisioni_al_secondo"] = round(3 * n / (sum(ms) / 1000), 2)
    s["token_generati_totali"] = sum(r.get("token_generati") or 0 for r in righe)
    s["errori"] = [f'{r["id"]} {k}: atteso {r["atteso"][k]}, dato {r["predetto"][k]}'
                   for r in righe for k in CAMPI if not giusto(r, k)]
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, default=Path(__file__).with_name("dataset") / "richieste.jsonl")
    ap.add_argument("--rizzo", help="URL del server rizzo-flow, es. http://127.0.0.1:8017")
    ap.add_argument("--ollama", metavar="MODELLO", help="modello Ollama per il confronto")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if bool(args.rizzo) == bool(args.ollama):
        ap.error("scegli un motore: --rizzo oppure --ollama")

    casi = [json.loads(l) for l in args.dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
    classifica = ((lambda t: rizzo(args.rizzo, t)) if args.rizzo
                  else (lambda t: ollama(args.ollama_url, args.ollama, t)))

    classifica(casi[0]["testo"])  # riscaldamento: carica modello e cache, escluso dalle misure
    righe = []
    for c in casi:
        pred, extra = classifica(c["testo"])
        righe.append({"id": c["id"], "atteso": {k: c[k] for k in CAMPI}, "predetto": pred, **extra})
        print(c["id"], " ".join(f"{k}={pred[k]}" for k in CAMPI), f'{extra["ms"]:.0f} ms', flush=True)

    s = riepilogo(righe)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "data": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "motore": "rizzo-flow" if args.rizzo else "ollama",
        "modello": righe[0].get("modello"), "riepilogo": s, "righe": righe,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items() if k != "errori"}, ensure_ascii=False, indent=1))
    print("errori:", *s["errori"] or ["nessuno"], sep="\n  ")


if __name__ == "__main__":
    main()
