import numpy as np 
import json 
from pathlib import Path

def general_category(category):
    if category == "vehicle.car":
        return "car"
    
    elif (
        category.startswith("vehicle.truck")
        or category.startswith("vehicle.bus")
        or category.startswith("vehicle.trailer")
    ):
        return "large_vehicle"

    elif (
        category.startswith("vehicle.bicycle")
        or category.startswith("vehicle.motorcycle")
    ):
        return "two_wheeler"
    
    elif category.startswith("human.pedestrian"):
        return "pedestrian"
    
    elif category.startswith("vehicle."): 
        return "other_vehicle"
    
    else: 
        return "other"