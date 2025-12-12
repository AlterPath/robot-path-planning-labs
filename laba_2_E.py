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

class DijkstraPathFinder:
    """Implementation of Dijkstra's algorithm"""
    
    def __init__(self, environment: GridEnvironment):
        self.env = environment
        self.allow_diagonal = True
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], Dict]:
        """Find path from start to goal"""
        if not self.env.is_valid(*start) or not self.env.is_valid(*goal):
            raise ValueError("Start or goal point is in obstacle")
        
        # Initialization
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        cost_so_far = defaultdict(lambda: float('inf'))
        cost_so_far[start] = 0
        visited_nodes = 0
        
        start_time = time.time()
        
        while open_set:
            current_cost, current = heapq.heappop(open_set)
            visited_nodes += 1
            
            # Check if goal reached
            if current == goal:
                break
            
            # Process neighbors
            for neighbor in self.env.get_neighbors(*current, self.allow_diagonal):
                new_cost = cost_so_far[current] + self.env.get_cost(*current, *neighbor)
                
                if new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(open_set, (new_cost, neighbor))
                    came_from[neighbor] = current
        
        # Reconstruct path
        path = []
        if goal in came_from or start == goal:
            current = goal
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
        
        elapsed_time = time.time() - start_time
        
        stats = {
            'time': elapsed_time,
            'visited_nodes': visited_nodes,
            'path_length': len(path) - 1 if path else 0,
            'total_cost': cost_so_far[goal] if goal in cost_so_far else float('inf')
        }
        
        return path, stats
    
    def get_visited_nodes(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get list of visited nodes (for visualization)"""
        _, stats = self.find_path(start, goal)
        # In real implementation, all visited nodes should be saved
        # Here we return path for simplicity
        path, _ = self.find_path(start, goal)
        return path

class AStarPathFinder:
    """Implementation of A* algorithm"""
    
    def __init__(self, environment: GridEnvironment):
        self.env = environment
        self.allow_diagonal = True
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Heuristic function (Manhattan distance)"""
        x1, y1 = a
        x2, y2 = b
        dx, dy = abs(x1 - x2), abs(y1 - y2)
        
        if self.allow_diagonal:
            # Diagonal distance (Chebyshev)
            return max(dx, dy)
        else:
            # Manhattan distance
            return dx + dy
    
    def heuristic_euclidean(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Heuristic function (Euclidean distance)"""
        x1, y1 = a
        x2, y2 = b
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                  heuristic_type: str = 'manhattan') -> Tuple[List[Tuple[int, int]], Dict]:
        """Find path from start to goal with selected heuristic"""
        if not self.env.is_valid(*start) or not self.env.is_valid(*goal):
            raise ValueError("Start or goal point is in obstacle")
        
        # Select heuristic function
        if heuristic_type == 'euclidean':
            heuristic_func = self.heuristic_euclidean
        else:
            heuristic_func = self.heuristic
        
        # Initialization
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start] = 0
        
        f_score = defaultdict(lambda: float('inf'))
        f_score[start] = heuristic_func(start, goal)
        
        visited_nodes = 0
        start_time = time.time()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            visited_nodes += 1
            
            if current == goal:
                break
            
            for neighbor in self.env.get_neighbors(*current, self.allow_diagonal):
                tentative_g_score = g_score[current] + self.env.get_cost(*current, *neighbor)
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic_func(neighbor, goal)
                    
                    # Check if neighbor is already in open_set
                    in_open_set = any(neighbor == item[1] for item in open_set)
                    if not in_open_set:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # Reconstruct path
        path = []
        if goal in came_from or start == goal:
            current = goal
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
        
        elapsed_time = time.time() - start_time
        
        stats = {
            'time': elapsed_time,
            'visited_nodes': visited_nodes,
            'path_length': len(path) - 1 if path else 0,
            'total_cost': g_score[goal] if goal in g_score else float('inf'),
            'heuristic_type': heuristic_type
        }
        
        return path, stats

class DStarPathFinder:
    """Simplified implementation of D* Lite algorithm"""
    
    def __init__(self, environment: GridEnvironment):
        self.env = environment
        self.allow_diagonal = True
        self.km = 0  # Key modifier
        self.U = []  # Priority queue
        self.rhs = defaultdict(lambda: float('inf'))
        self.g = defaultdict(lambda: float('inf'))
        self.backpointers = {}
        
    def calculate_key(self, node: Tuple[int, int], start: Tuple[int, int]) -> Tuple[float, float]:
        """Calculate key for priority queue"""
        g_rhs_min = min(self.g[node], self.rhs[node])
        return (g_rhs_min + self.heuristic(start, node) + self.km, g_rhs_min)
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Heuristic function"""
        x1, y1 = a
        x2, y2 = b
        dx, dy = abs(x1 - x2), abs(y1 - y2)
        return math.sqrt(dx*dx + dy*dy)
    
    def update_vertex(self, u: Tuple[int, int], start: Tuple[int, int], goal: Tuple[int, int]):
        """Update vertex"""
        if u != goal:
            # Calculate minimum cost among neighbors
            min_rhs = float('inf')
            best_successor = None
            
            for v in self.env.get_neighbors(*u, self.allow_diagonal):
                cost = self.env.get_cost(*u, *v)
                candidate_rhs = self.g[v] + cost
                
                if candidate_rhs < min_rhs:
                    min_rhs = candidate_rhs
                    best_successor = v
            
            self.rhs[u] = min_rhs
            self.backpointers[u] = best_successor
        
        # Remove from queue if present
        self.U = [item for item in self.U if item[1] != u]
        
        # Add back if g != rhs
        if self.g[u] != self.rhs[u]:
            heapq.heappush(self.U, (self.calculate_key(u, start), u))
    
    def compute_shortest_path(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """Main procedure for computing shortest path"""
        while self.U and (self.U[0][0] < self.calculate_key(start, start) or 
                         self.rhs[start] != self.g[start]):
            k_old, u = heapq.heappop(self.U)
            k_new = self.calculate_key(u, start)
            
            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in self.env.get_neighbors(*u, self.allow_diagonal):
                    self.update_vertex(s, start, goal)
            else:
                self.g[u] = float('inf')
                for s in self.env.get_neighbors(*u, self.allow_diagonal) + [u]:
                    self.update_vertex(s, start, goal)
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], Dict]:
        """Find path from start to goal with replanning capability"""
        if not self.env.is_valid(*start) or not self.env.is_valid(*goal):
            raise ValueError("Start or goal point is in obstacle")
        
        # Initialization
        self.km = 0
        self.U = []
        self.rhs = defaultdict(lambda: float('inf'))
        self.g = defaultdict(lambda: float('inf'))
        self.backpointers = {}
        
        # Goal initialization
        self.rhs[goal] = 0
        self.U.append((self.calculate_key(goal, start), goal))
        heapq.heapify(self.U)
        
        visited_nodes = 0
        start_time = time.time()
        
        # Initial planning
        self.compute_shortest_path(start, goal)
        visited_nodes = len(self.g)
        
        # Reconstruct path
        path = []
        current = start
        
        while current != goal:
            path.append(current)
            if current not in self.backpointers:
                break
            current = self.backpointers[current]
            
            # Limit path length to avoid infinite loops
            if len(path) > self.env.width * self.env.height:
                break
        
        if current == goal:
            path.append(goal)
        
        elapsed_time = time.time() - start_time
        
        stats = {
            'time': elapsed_time,
            'visited_nodes': visited_nodes,
            'path_length': len(path) - 1 if len(path) > 1 else 0,
            'total_cost': self.g[start] if start in self.g else float('inf')
        }
        
        return path, stats
    
    def replan(self, start: Tuple[int, int], goal: Tuple[int, int], 
               changed_cells: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], Dict]:
        """Replan path when environment changes"""
        # Update cost of changed cells
        for cell in changed_cells:
            # Here should be logic for updating cost
            # In simplified implementation, we just restart the search
            pass
        
        # Increase km to account for environment changes
        self.km += self.heuristic(start, goal)
        
        # Replanning
        return self.find_path(start, goal)

class PathPlanningBenchmark:
    """Class for comparative analysis of path planning algorithms"""
    
    def __init__(self, env_width: int = 50, env_height: int = 50):
        self.env = GridEnvironment(env_width, env_height)
        self.results = {}
        
    def setup_test_environment(self, test_case: str = "medium"):
        """Set up test environment"""
        # Clear environment
        self.env.grid = np.zeros((self.env.height, self.env.width), dtype=np.float32)
        self.env.obstacles = []
        
        if test_case == "simple":
            # Simple environment
            self.env.add_rectangle_obstacle(20, 20, 30, 30)
            
        elif test_case == "medium":
            # Medium complexity
            self.env.add_rectangle_obstacle(10, 10, 20, 40)
            self.env.add_rectangle_obstacle(30, 30, 40, 45)
            self.env.add_circular_obstacle(35, 15, 8)
            
        elif test_case == "complex":
            # Complex environment
            self.env.add_random_obstacles(15, 4, 10)
            self.env.add_rectangle_obstacle(0, 0, 50, 5)  # bottom wall
            self.env.add_rectangle_obstacle(0, 45, 50, 50)  # top wall
            self.env.add_rectangle_obstacle(0, 0, 5, 50)  # left wall
            self.env.add_rectangle_obstacle(45, 0, 50, 50)  # right wall
    
    def run_benchmark(self, start: Tuple[int, int], goal: Tuple[int, int],
                     algorithms: List[str] = ['dijkstra', 'astar', 'dstar']):
        """Run benchmark for specified algorithms"""
        
        self.results = {}
        
        # Test each algorithm
        if 'dijkstra' in algorithms:
            print("\n" + "="*50)
            print("Testing Dijkstra's Algorithm")
            print("="*50)
            
            dijkstra = DijkstraPathFinder(self.env)
            dijkstra_path, dijkstra_stats = dijkstra.find_path(start, goal)
            self.results['dijkstra'] = {
                'path': dijkstra_path,
                'stats': dijkstra_stats
            }
            print(f"Time: {dijkstra_stats['time']:.4f}s")
            print(f"Visited nodes: {dijkstra_stats['visited_nodes']}")
            print(f"Path length: {dijkstra_stats['path_length']}")
            print(f"Total cost: {dijkstra_stats['total_cost']:.2f}")
        
        if 'astar' in algorithms:
            print("\n" + "="*50)
            print("Testing A* Algorithm")
            print("="*50)
            
            astar = AStarPathFinder(self.env)
            
            # Test different heuristics
            for heuristic_type in ['manhattan', 'euclidean']:
                print(f"\nA* with {heuristic_type} heuristic:")
                astar_path, astar_stats = astar.find_path(start, goal, heuristic_type)
                self.results[f'astar_{heuristic_type}'] = {
                    'path': astar_path,
                    'stats': astar_stats
                }
                print(f"Time: {astar_stats['time']:.4f}s")
                print(f"Visited nodes: {astar_stats['visited_nodes']}")
                print(f"Path length: {astar_stats['path_length']}")
                print(f"Total cost: {astar_stats['total_cost']:.2f}")
        
        if 'dstar' in algorithms:
            print("\n" + "="*50)
            print("Testing D* Lite Algorithm")
            print("="*50)
            
            dstar = DStarPathFinder(self.env)
            dstar_path, dstar_stats = dstar.find_path(start, goal)
            self.results['dstar'] = {
                'path': dstar_path,
                'stats': dstar_stats
            }
            print(f"Time: {dstar_stats['time']:.4f}s")
            print(f"Visited nodes: {dstar_stats['visited_nodes']}")
            print(f"Path length: {dstar_stats['path_length']}")
            print(f"Total cost: {dstar_stats['total_cost']:.2f}")
        
        return self.results
    
    def compare_results(self):
        """Compare and display benchmark results"""
        if not self.results:
            print("No results to compare. Run benchmark first.")
            return
        
        print("\n" + "="*60)
        print("COMPARATIVE ANALYSIS OF PATH PLANNING ALGORITHMS")
        print("="*60)
        
        # Prepare data for comparison
        comparison_data = []
        
        for algo_name, data in self.results.items():
            stats = data['stats']
            comparison_data.append({
                'Algorithm': algo_name,
                'Time (s)': f"{stats['time']:.4f}",
                'Visited Nodes': stats['visited_nodes'],
                'Path Length': stats['path_length'],
                'Total Cost': f"{stats['total_cost']:.2f}" if stats['total_cost'] != float('inf') else "N/A"
            })
        
        # Display comparison table
        headers = ['Algorithm', 'Time (s)', 'Visited Nodes', 'Path Length', 'Total Cost']
        row_format = "{:<20} {:<10} {:<15} {:<12} {:<10}"
        
        print("\n" + row_format.format(*headers))
        print("-" * 70)
        
        for row in comparison_data:
            print(row_format.format(
                row['Algorithm'],
                row['Time (s)'],
                row['Visited Nodes'],
                row['Path Length'],
                row['Total Cost']
            ))
    
    def visualize_comparison(self):
        """Visualize comparison of different algorithms"""
        if not self.results:
            print("No results to visualize. Run benchmark first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Path Planning Algorithms Comparison', fontsize=16, y=1.02)
        
        # Environment visualization
        self.env.visualize(title="Environment with obstacles")
        
        # Create separate figure for algorithm comparison
        fig2, ax_compare = plt.subplots(figsize=(10, 8))
        
        # Prepare data for bar charts
        algorithms = []
        times = []
        visited_nodes = []
        
        for algo_name, data in self.results.items():
            algorithms.append(algo_name.replace('_', ' ').title())
            times.append(data['stats']['time'])
            visited_nodes.append(data['stats']['visited_nodes'])
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        # Time comparison
        ax_compare.bar(x - width/2, times, width, label='Time (s)', color='skyblue')
        
        # Visited nodes comparison (secondary axis)
        ax2 = ax_compare.twinx()
        ax2.bar(x + width/2, visited_nodes, width, label='Visited Nodes', color='lightcoral')
        
        ax_compare.set_xlabel('Algorithm')
        ax_compare.set_ylabel('Time (seconds)', color='skyblue')
        ax2.set_ylabel('Visited Nodes', color='lightcoral')
        ax_compare.set_title('Performance Comparison')
        ax_compare.set_xticks(x)
        ax_compare.set_xticklabels(algorithms, rotation=45, ha='right')
        
        # Add legend
        lines1, labels1 = ax_compare.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_compare.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def run_comprehensive_test(self, num_runs: int = 10):
        """Run comprehensive tests with multiple scenarios"""
        test_cases = ['simple', 'medium', 'complex']
        
        summary_results = {}
        
        for test_case in test_cases:
            print(f"\n{'='*60}")
            print(f"Running tests for {test_case.upper()} environment")
            print('='*60)
            
            # Setup environment
            self.setup_test_environment(test_case)
            
            # Define start and goal positions
            start = (5, 5)
            goal = (45, 45)
            
            # Run benchmark
            case_results = self.run_benchmark(
                start, goal,
                algorithms=['dijkstra', 'astar']
            )
            
            summary_results[test_case] = case_results
            
            # Visualize one path
            if test_case == 'medium':  # Visualize only for medium case
                dijkstra_path = case_results['dijkstra']['path']
                self.env.visualize(dijkstra_path, f"Dijkstra - {test_case.title()} Environment")
                plt.show()
        
        return summary_results