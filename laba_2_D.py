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