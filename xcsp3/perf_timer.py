from __future__ import annotations
import contextvars
from statistics import mean
from typing import List, Union, Dict, Optional
import time, math, json

import os, sys, pathlib

sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), ".."))
from xcsp3 import __perf_context, __timer_context

__timer_context.set([])

def get_perf_context() -> PerfContext:
    try:
        return __perf_context.get()
    except LookupError:
        return None
    
def set_perf_context(context:Optional[PerfContext]):
    __perf_context.set(context)

def get_timer_context(level:int=0) -> TimerContext:
    try:
        return __timer_context.get()[0]
    except LookupError:
        return None
    
def set_timer_context(context:Optional[TimerContext]):
    timer_context = __timer_context.get()
    timer_context.append(context)
    __timer_context.set(timer_context)
    return len(timer_context)-1

class PerfContext:
    name:str = "no_name"
    label_counter: int = 0
    measurements = {}

    def __init__(self, path=None):
        self.path = path

    def __enter__(self):
        try:
            self.measurements = {}
            set_perf_context(self)
        except:
            print("ERROR")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        set_perf_context(None)
        return False

    def get_label(self):
        label = self.label_counter
        self.label_counter += 1
        return str(label)
    
    def add_time_measurement(self, label, time):
        self.measurements[label] = self.measurements.get(label, []) + [time] 

        if self.path is not None:
            with open(self.path, 'w') as f:
                json.dump(self.measurements, f, indent=4)

    def aggregate(self):
        for k,v in self.measurements.items():
            print(k, ":", mean(v))

class TimerContext:
    name:str = "no_name"
    label:str
    start: float
    end: float

    def __init__(self, label:Optional[str]):
        if label is None:
            perf_context = get_perf_context()
            label = perf_context.get_label()
        self.label = label

    def __enter__(self):
        self.level = set_timer_context(self)
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
    
        perf_context = get_perf_context()
        self.end = time.time()
        set_timer_context(None)
        
        # try:
        perf_context.add_time_measurement(self.label, self.end - self.start)
        # except:
        #     pass

    @property
    def time(self):
        return self.end - self.start

    def to_dict(self):
        return {self.label: self.end - self.start}
