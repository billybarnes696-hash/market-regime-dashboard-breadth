
from __future__ import annotations
from typing import Optional, Dict, Tuple
import numpy as np

def ema(prev: float, new: float, alpha: float) -> float:
    return alpha * new + (1.0 - alpha) * prev

def mcclellan_oscillator_from_feed(adv_issues: float, dec_issues: float, ema19_prev: float, ema39_prev: float, ratio_adjust: bool=True) -> Tuple[float, float, float]:
    total = adv_issues + dec_issues
    if total <= 0:
        net = 0.0
    else:
        raw = adv_issues - dec_issues
        net = (1000.0 * raw / total) if ratio_adjust else raw
    alpha19 = 2.0 / 20.0
    alpha39 = 2.0 / 40.0
    ema19_now = ema(ema19_prev, net, alpha19)
    ema39_now = ema(ema39_prev, net, alpha39)
    return float(ema19_now - ema39_now), float(ema19_now), float(ema39_now)

def nysi_from_nymo(nysi_prev: float, nymo_now: float) -> float:
    return float(nysi_prev + nymo_now)

def cpce_from_feed(put_volume: float, call_volume: float) -> Optional[float]:
    if call_volume <= 0:
        return None
    return float(put_volume / call_volume)

def nymo_proxy_from_features(feature_values: Dict[str, float], weights: Dict[str, float]) -> float:
    total = 0.0
    for k, v in feature_values.items():
        total += weights.get(k, 0.0) * v * 1000.0
    return float(np.clip(total, -120.0, 120.0))

def nysi_proxy_from_nymo(nysi_prev: float, nymo_proxy_now: float, scale: float=0.35) -> float:
    return float(nysi_prev + (nymo_proxy_now * scale))

def cpce_proxy_from_stress(vxx_ret: float, spxs_ret: float, svol_ret: float) -> float:
    stress = (1.6 * vxx_ret) + (0.8 * spxs_ret) - (0.8 * svol_ret)
    cpce_proxy = 0.60 + (stress * 10.0)
    return float(np.clip(cpce_proxy, 0.35, 1.20))
