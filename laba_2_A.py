import numpy as np
import matplotlib.pyplot as plt
import heapq
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import math

@dataclass(order=True)
class Node:
    """Graph node for pathfinding algorithms"""
    x: int = field(compare=False)
    y: int = field(compare=False)
    cost: float = field(compare=False)
    priority: float = field(compare=True)  # For priority queue
    
    def __init__(self, x, y, cost=float('inf'), priority=float('inf')):
        self.x = x
        self.y = y
        self.cost = cost
        self.priority = priority

class GridEnvironment:
    """Class representing a grid-based environment"""
    
    def __init__(self, width: int = 50, height: int = 50):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.float32)
        self.obstacles = []
        
    def add_rectangle_obstacle(self, x1: int, y1: int, x2: int, y2: int):
        """Add rectangular obstacle"""
        self.grid[y1:y2, x1:x2] = 1.0
        self.obstacles.append(('rect', (x1, y1, x2, y2)))
    
    def add_circular_obstacle(self, cx: int, cy: int, radius: int):
        """Add circular obstacle"""
        for y in range(max(0, cy - radius), min(self.height, cy + radius)):
            for x in range(max(0, cx - radius), min(self.width, cx + radius)):
                if math.sqrt((x - cx)**2 + (y - cy)**2) <= radius:
                    self.grid[y, x] = 1.0
        self.obstacles.append(('circle', (cx, cy, radius)))
    
    def add_random_obstacles(self, num_obstacles: int, min_size: int = 3, max_size: int = 7):
        """Add random obstacles"""
        for _ in range(num_obstacles):
            x = np.random.randint(0, self.width - max_size)
            y = np.random.randint(0, self.height - max_size)
            width = np.random.randint(min_size, max_size)
            height = np.random.randint(min_size, max_size)
            self.add_rectangle_obstacle(x, y, x + width, y + height)
    
    def is_valid(self, x: int, y: int) -> bool:
        """Check if cell is traversable"""
        return (0 <= x < self.width and 0 <= y < self.height 
                and self.grid[y, x] == 0)
    
    def get_neighbors(self, x: int, y: int, allow_diagonal: bool = True) -> List[Tuple[int, int]]:
        """Get neighboring cells"""
        neighbors = []
        
        # 4-connected neighbors
        directions_4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # 8-connected neighbors (including diagonals)
        directions_8 = [(0, 1), (1, 0), (0, -1), (-1, 0),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        directions = directions_8 if allow_diagonal else directions_4
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def get_cost(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Get movement cost between cells"""
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        
        # Diagonal movement costs more
        if dx > 0 and dy > 0:
            return math.sqrt(2)  # ≈1.414
        else:
            return 1.0
    
    def visualize(self, path: List[Tuple[int, int]] = None, 
                  title: str = "Environment with obstacles"):
        """Visualize environment and path"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Display grid
        cmap = plt.cm.Greys
        cmap.set_bad(color='white')
        
        # Create mask for obstacles
        display_grid = np.ma.masked_where(self.grid == 1, self.grid)
        
        ax.imshow(display_grid, cmap=cmap, origin='lower', 
                  extent=[0, self.width, 0, self.height])
        
        # Display path
        if path:
            path_array = np.array(path)
            ax.plot(path_array[:, 0] + 0.5, path_array[:, 1] + 0.5, 
                   'r-', linewidth=2, label='Path')
            ax.scatter(path_array[0, 0] + 0.5, path_array[0, 1] + 0.5,
                      c='green', s=200, marker='o', label='Start')
            ax.scatter(path_array[-1, 0] + 0.5, path_array[-1, 1] + 0.5,
                      c='red', s=200, marker='o', label='Goal')
        
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig, ax