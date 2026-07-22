# PRT — Architecture des building blocks

```mermaid
flowchart TB
    BBG["🖥️ Bloomberg Terminal<br/>(xbbg / blpapi)"]

    subgraph DATA["prt/data — Ingestion (seul bloc qui parle à Bloomberg)"]
        direction TB
        IFACE["Interface DataProvider"]
        BBGP["BloombergProvider<br/>bdh / bdp (lazy import)"]
        STREAM["StreamDaemon<br/>blp.live → LAST_PRICE, BID, ASK<br/>+ reconnexion, fallback bdp"]
        MOCK["MockProvider<br/>données synthétiques (CI, dev sans terminal)"]
        JOBS["Jobs quotidiens<br/>top-up historique · détection de roll<br/>→ full refresh si contrat actif change"]
        IFACE --- BBGP
        IFACE --- MOCK
        BBGP --- STREAM
        JOBS --> IFACE
    end

    subgraph DB["prt/db — SQLite (source de vérité unique)"]
        direction LR
        T1["Market data<br/>prices_series (roll-adj / CR)<br/>prices_generic_raw (G1, G2)<br/>active_contracts · live_quotes"]
        T2["Référentiel<br/>instruments<br/>(multiplier, ccy, coûts,<br/>règle de roll, close time)"]
        T3["Signaux & trading<br/>forecasts · targets<br/>fills · positions · pnl"]
        T4["Ops<br/>refresh_log · bbg_query_log<br/>eco_releases (phase 2)"]
    end

    CONFIG["⚙️ config/ (YAML)<br/>univers · capital 100M · vol cible 10%<br/>poids stratégies · coûts · conventions"]

    subgraph SIG["prt/signals — Moteur de signaux"]
        direction TB
        PIT["DataView point-in-time<br/>ligne t = info connaissable avant close t<br/>(prix ≤ t−1, calendrier ≤ t)<br/>⚠️ lag dans l'accès data, jamais de shift"]
        BASE["Signal (classe de base)<br/>universe · requires · compute → forecast ∈ [−5, +5]"]
        S1["momentum"]
        S2["carry"]
        S3["seaso · events…<br/>(plug-in : 1 fichier = 1 signal)"]
        LAB["🔬 Signal Lab<br/>corrélation des forecasts<br/>corrélation des PnL subsystems<br/>Sharpe marginal d'un candidat"]
        PIT --> BASE
        BASE --- S1
        BASE --- S2
        BASE --- S3
    end

    subgraph PORT["prt/portfolio — Signal → Position"]
        direction TB
        COMB["Combinaison<br/>poids × forecasts · divers. multiplier · cap"]
        VOLT["Vol targeting<br/>EWMA vol instrument → notionnel cible"]
        TRAD["Traduction<br/>futures : notionnel → contrats entiers<br/>FX : notionnel de la paire"]
        BUF["Buffering anti-churn"]
        COMB --> VOLT --> TRAD --> BUF
    end

    subgraph BT["prt/backtest"]
        SIM["Simulateur daily<br/>signal(close t−1) → fill au close t<br/>+ demi-spread par ticker"]
        REP["Rapports<br/>PnL par signal / instrument / stratégie<br/>stats, drawdowns"]
        SIM --> REP
    end

    subgraph LIVE["prt/live"]
        PNL["PnL depuis dernier close<br/>position × Δprix × mult × FX"]
        PREV["Preview<br/>« si ça clôture ici,<br/>target demain = X »"]
        ORD["Génération d'ordres<br/>targets vs book (buffer)"]
        FILLS["Saisie des fills<br/>(manuelle)"]
    end

    subgraph DASH["prt/dashboard — Streamlit (localhost)"]
        P1["PnL live"]
        P2["Positions & risque<br/>contrats · notionnel · contrib vol"]
        P3["Signaux & preview"]
        P4["Ordres"]
        P5["Corrélations & backtests"]
        P6["Santé data<br/>âge des quotes · quota BBG · rolls à venir"]
    end

    TESTS["✅ CI GitHub Actions<br/>lint + tests par bloc, MockProvider + fixtures<br/>zéro dépendance terminal<br/>(tests @requires_terminal skippés)"]

    BBG -->|"historique + référentiel + stream"| DATA
    DATA -->|"écrit"| DB
    CONFIG --> DB
    CONFIG -.-> SIG
    CONFIG -.-> PORT
    DB -->|"vue point-in-time"| SIG
    SIG -->|"forecasts"| DB
    DB -->|"forecasts + vols + réf"| PORT
    PORT -->|"targets"| DB
    DB --> BT
    DB --> LIVE
    LIVE -->|"ordres → exécution manuelle → fills"| DB
    DB -->|"lecture seule"| DASH
    LAB -.->|"lit forecasts & pnl"| DB
```

## Règles de dépendance

1. **Bloomberg n'est touché que par `prt/data`** — tout le reste lit la DB. Le repo tourne intégralement sans terminal (MockProvider).
2. **La DB est le seul point de rencontre** — aucun bloc ne parle latéralement à un autre ; chaque bloc est autonome, testable seul, remplaçable.
3. **Backtest et live exécutent le même code** signaux + portfolio — seule la source des dates change.
4. **Le lag d'information vit dans la DataView** (knowledge time par type de donnée), jamais en aval — pas de `shift(1)` mécanique.
