from dataclasses import dataclass, field
from typing import Sequence, Tuple


# All config in one file

VOL_TARGET = 0.15
VOL_LOOKBACK = 60          # EWM span for ex-ante daily vol
RETURN_HORIZONS = (1, 21, 63, 126, 252)
MACD_PAIRS: Tuple[Tuple[int, int], ...] = ((8, 24), (16, 28), (32, 96))
NUM_FEATURES = len(RETURN_HORIZONS) + len(MACD_PAIRS)  
EPS = 1e-8

# default ETF universe
DEFAULT_TICKERS = [
    # US equity
    "SPY", "QQQ", "IWM", "VTI",
    # International equity
    "EFA", "EEM",
    # US sector equity
    "XLF", "XLE", "XLK", "XLI", "XLP", "XLV",
    # Real estate
    "VNQ",
    # Rates / fixed income
    "TLT", "IEF", "SHY", "LQD", "HYG",
    # Commodities / FX
    "GLD", "DBC", "UUP",
]


@dataclass
class DataConfig:
    tickers: Sequence[str] = None          # None → DEFAULT_TICKERS
    start: str = "2005-01-01"
    end: str = "2025-12-31"
    train_frac: float = 0.70
    val_frac: float = 0.15
    lookback: int = 126                    # l_t in the paper
    context_len: int = 21                  # l_c for X-Trend
    num_context: int = 10                  # |C|
    batch_size: int = 64
    seed: int = 42

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = list(DEFAULT_TICKERS)


@dataclass
class ModelConfig:
    """Shared between baseline and X-Trend."""
    hidden_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.1
    warmup_steps: int = 63                 # l_s – ignore first predictions


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    epochs: int = 30
    patience: int = 10
