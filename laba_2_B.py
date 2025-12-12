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