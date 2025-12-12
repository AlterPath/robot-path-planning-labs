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