"""PRT — systematic macro fund infrastructure.

Building blocks (each self-contained, DB is the only meeting point):
    prt.config      configuration & instrument master
    prt.db          SQLite persistence layer
    prt.data        Bloomberg / mock data providers, ingestion, live stream
    prt.signals     point-in-time DataView, signal engine, signal lab
    prt.portfolio   forecast -> position translation (vol targeting)
    prt.backtest    daily simulator + reports
    prt.live        intraday PnL, signal preview, order generation
    prt.dashboard   Streamlit UI (read-only on the DB)
"""

__version__ = "0.1.0"
