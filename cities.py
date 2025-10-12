import heapq
from collections import defaultdict
from typing import List, Tuple, Dict, Set


def build_graph(edges: List[List[int]]) -> Dict[int, List[Tuple[int, int]]]:
    graph = defaultdict(list)
    
    for city_a, city_b, distance in edges:
        graph[city_a].append((city_b, distance))
        graph[city_b].append((city_a, distance))
    
    return dict(graph)


def dijkstra(graph: Dict[int, List[Tuple[int, int]]], start_city: int) -> Dict[int, int]:
    distances = {city: float('inf') for city in graph}
    distances[start_city] = 0
    
    heap = [(0, start_city)]
    visited = set()
    
    while heap:
        current_distance, current_city = heapq.heappop(heap)
        
        if current_city in visited:
            continue
            
        visited.add(current_city)
        
        for neighbor, edge_distance in graph.get(current_city, []):
            if neighbor in visited:
                continue
                
            new_distance = current_distance + edge_distance

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(heap, (new_distance, neighbor))
    
    return distances


def count_reachable_cities(graph: Dict[int, List[Tuple[int, int]]], 
                          start_city: int, 
                          mileage: int) -> int:
    distances = dijkstra(graph, start_city)

    reachable_count = 0
    for city, distance in distances.items():
        if distance <= mileage:
            reachable_count += 1
    
    return reachable_count


def find_city_with_min_reachable(graph: Dict[int, List[Tuple[int, int]]], 
                                mileage: int) -> Tuple[int, int]:
    min_reachable = float('inf')
    best_city = None

    for city in graph:
        reachable_count = count_reachable_cities(graph, city, mileage)
        
        if reachable_count < min_reachable:
            min_reachable = reachable_count
            best_city = city
    
    return best_city, min_reachable


def solve_cities_problem(edges: List[List[int]], mileage: int) -> Tuple[int, int]:
    if not edges:
        return None, 0

    graph = build_graph(edges)
    
    if not graph:
        return None, 0
    
    return find_city_with_min_reachable(graph, mileage)


def print_graph_info(graph: Dict[int, List[Tuple[int, int]]]):
    """Выводит информацию о графе."""
    print("Граф городов:")
    for city, neighbors in graph.items():
        neighbor_str = ", ".join([f"{neighbor}({dist})" for neighbor, dist in neighbors])
        print(f"  Город {city}: {neighbor_str}")


def print_distances_from_city(graph: Dict[int, List[Tuple[int, int]]], 
                             start_city: int, 
                             mileage: int):
    """Выводит информацию о расстояниях от указанного города."""
    distances = dijkstra(graph, start_city)
    reachable_cities = []
    
    print(f"\nРасстояния от города {start_city}:")
    for city, distance in sorted(distances.items()):
        status = "yes" if distance <= mileage else "no"
        print(f"  До города {city}: {distance} {status}")
        if distance <= mileage:
            reachable_cities.append(city)
    
    print(f"Доступных городов в пределах {mileage}: {len(reachable_cities)}")
    print(f"Список доступных городов: {sorted(reachable_cities)}")


if __name__ == "__main__":

    edges1 = [
        [1, 2, 10],  
        [2, 3, 15],  
        [1, 3, 30],  
        [3, 4, 5],   
        [2, 4, 20],  
    ]
    mileage1 = 25
    
    print(f"Рёбра: {edges1}")
    print(f"Максимальное расстояние: {mileage1}")
    
    graph1 = build_graph(edges1)
    print_graph_info(graph1)
    
    best_city, min_reachable = solve_cities_problem(edges1, mileage1)
    print(f"\nРезультат:")
    print(f"Лучший город: {best_city}")
    print(f"Минимальное количество доступных городов: {min_reachable}")

    print_distances_from_city(graph1, best_city, mileage1)

    for city in graph1:
        reachable_count = count_reachable_cities(graph1, city, mileage1)
        print(f"Город {city}: {reachable_count} доступных городов")
    