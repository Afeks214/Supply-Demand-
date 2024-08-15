import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum

class FVGType(Enum):
    BULLISH = 1
    BEARISH = 2

@dataclass
class FVG:
    max: float
    min: float
    type: FVGType
    time: pd.Timestamp
    touched: bool = False

class FairValueGap:
    def __init__(self, threshold_per: float = 0, auto: bool = False,
                 show_last: int = 0, mitigation_levels: bool = False,
                 extend: int = 20, dynamic: bool = False,
                 bull_css: str = '#089981', bear_css: str = '#f23645'):
        self.threshold_per = threshold_per / 100
        self.auto = auto
        self.show_last = show_last
        self.mitigation_levels = mitigation_levels
        self.extend = extend
        self.dynamic = dynamic
        self.bull_css = bull_css
        self.bear_css = bear_css
        
        self.fvg_records: List[FVG] = []
        self.bull_count = self.bear_count = self.bull_mitigated = self.bear_mitigated = 0
        self.max_bull_fvg = self.min_bull_fvg = self.max_bear_fvg = self.min_bear_fvg = np.nan

    def detect_fvg(self, data: pd.DataFrame) -> Tuple[bool, bool, Optional[FVG]]:
        if self.auto:
            threshold = (data['high'] - data['low']).cumsum().iloc[-1] / len(data)
        else:
            threshold = self.threshold_per

        bull_fvg = (data['low'].iloc[-1] > data['high'].iloc[-3]) and \
                   (data['close'].iloc[-2] > data['high'].iloc[-3]) and \
                   ((data['low'].iloc[-1] - data['high'].iloc[-3]) / data['high'].iloc[-3] > threshold)
        
        bear_fvg = (data['high'].iloc[-1] < data['low'].iloc[-3]) and \
                   (data['close'].iloc[-2] < data['low'].iloc[-3]) and \
                   ((data['low'].iloc[-3] - data['high'].iloc[-1]) / data['high'].iloc[-1] > threshold)
        
        new_fvg = None
        if bull_fvg:
            new_fvg = FVG(data['low'].iloc[-1], data['high'].iloc[-3], FVGType.BULLISH, data.index[-1])
        elif bear_fvg:
            new_fvg = FVG(data['low'].iloc[-3], data['high'].iloc[-1], FVGType.BEARISH, data.index[-1])
        
        return bull_fvg, bear_fvg, new_fvg

    def process_fvgs(self, data: pd.DataFrame):
        bull_fvg, bear_fvg, new_fvg = self.detect_fvg(data)
        
        if new_fvg and (not self.fvg_records or new_fvg.time != self.fvg_records[-1].time):
            if self.dynamic:
                if new_fvg.type == FVGType.BULLISH:
                    self.max_bull_fvg, self.min_bull_fvg = new_fvg.max, new_fvg.min
                else:
                    self.max_bear_fvg, self.min_bear_fvg = new_fvg.max, new_fvg.min
            
            self.fvg_records.insert(0, new_fvg)
            if new_fvg.type == FVGType.BULLISH:
                self.bull_count += 1
            else:
                self.bear_count += 1
        elif self.dynamic:
            current_price = data['close'].iloc[-1]
            if bull_fvg:
                self.max_bull_fvg = max(min(current_price, self.max_bull_fvg), self.min_bull_fvg)
            elif bear_fvg:
                self.min_bear_fvg = min(max(current_price, self.min_bear_fvg), self.max_bear_fvg)
        
        self.check_mitigation(data['close'].iloc[-1])
        self.check_touched_fvgs(data)

    def check_mitigation(self, current_price: float):
        for fvg in self.fvg_records:
            if fvg.type == FVGType.BULLISH and current_price < fvg.min:
                self.bull_mitigated += 1
                self.fvg_records.remove(fvg)
            elif fvg.type == FVGType.BEARISH and current_price > fvg.max:
                self.bear_mitigated += 1
                self.fvg_records.remove(fvg)

    def check_touched_fvgs(self, data: pd.DataFrame):
        for fvg in self.fvg_records:
            if not fvg.touched:
                if fvg.type == FVGType.BULLISH and data['low'].iloc[-1] <= fvg.max:
                    fvg.touched = True
                elif fvg.type == FVGType.BEARISH and data['high'].iloc[-1] >= fvg.min:
                    fvg.touched = True

    def get_active_fvgs(self) -> List[FVG]:
        return self.fvg_records[:self.show_last] if self.show_last > 0 else self.fvg_records

    def get_touched_fvgs(self) -> List[FVG]:
        return [fvg for fvg in self.fvg_records if fvg.touched]

    def get_dynamic_fvgs(self) -> Dict[str, Optional[float]]:
        if self.dynamic:
            return {
                "max_bull_fvg": self.max_bull_fvg,
                "min_bull_fvg": self.min_bull_fvg,
                "max_bear_fvg": self.max_bear_fvg,
                "min_bear_fvg": self.min_bear_fvg
            }
        return {"max_bull_fvg": None, "min_bull_fvg": None, "max_bear_fvg": None, "min_bear_fvg": None}

    def get_stats(self) -> Dict[str, int]:
        return {
            "bull_count": self.bull_count,
            "bear_count": self.bear_count,
            "bull_mitigated": self.bull_mitigated,
            "bear_mitigated": self.bear_mitigated
        }

    def update(self, new_data: pd.DataFrame) -> Tuple[List[FVG], List[FVG], Dict[str, Optional[float]], Dict[str, int]]:
        self.process_fvgs(new_data)
        return (self.get_active_fvgs(), self.get_touched_fvgs(), 
                self.get_dynamic_fvgs(), self.get_stats())

# Example usage
if __name__ == "__main__":
    # Sample data
    data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106],
        'high': [102, 103, 104, 105, 106, 107, 108],
        'low':  [99, 100, 101, 102, 103, 104, 105],
        'close': [101, 102, 103, 104, 105, 106, 107]
    }, index=pd.date_range(start='2023-01-01', periods=7))

    fvg_indicator = FairValueGap(threshold_per=0.5, auto=True, show_last=3, dynamic=True)
    active_fvgs, touched_fvgs, dynamic_fvgs, stats = fvg_indicator.update(data)
    
    print("Active FVGs:", active_fvgs)
    print("Touched FVGs:", touched_fvgs)
    print("Dynamic FVGs:", dynamic_fvgs)
    print("Stats:", stats)
