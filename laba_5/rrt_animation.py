import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
import numpy as np

class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None
        self.children = []

class RRT:
    def __init__(self, start, goal, obstacles, area_bounds, max_iter=1000, step_size=10):
        self.start = Vertex(start[0], start[1])
        self.goal = Vertex(goal[0], goal[1])
        self.obstacles = obstacles
        self.area_bounds = area_bounds
        self.max_iter = max_iter
        self.step_size = step_size
        self.vertices = [self.start]
        
        # Для анимации
        self.fig, self.ax = None, None
        self.animation_vertices = []
        self.animation_edges = []
        self.animation_paths = []
        self.current_iteration = 0
        
    def plan_with_animation(self, save_animation=False):
        """Планирование с сохранением истории для анимации"""
        self.animation_vertices = [([self.start.x], [self.start.y])]
        self.animation_edges = []
        self.animation_paths = []
        self.current_iteration = 0
        
        for i in range(self.max_iter):
            self.current_iteration = i
            
            # Случайная точка
            rand_point = self.generate_random_point()
            
            # Ближайшая вершина
            nearest_vertex = self.find_nearest_vertex(rand_point)
            
            # Новая вершина
            new_vertex = self.steer(nearest_vertex, rand_point)
            
            # Проверка столкновений
            if self.is_collision_free(nearest_vertex, new_vertex):
                # В базовом RRT просто добавляем вершину к ближайшему родителю
                new_vertex.parent = nearest_vertex
                new_vertex.cost = nearest_vertex.cost + self.distance(nearest_vertex, new_vertex)
                self.vertices.append(new_vertex)
                
                # Сохраняем новое ребро для анимации
                if new_vertex.parent:
                    self.animation_edges.append(
                        ([new_vertex.parent.x, new_vertex.x], 
                         [new_vertex.parent.y, new_vertex.y])
                    )
                
                # Сохраняем текущее состояние дерева
                current_vertices_x = [v.x for v in self.vertices]
                current_vertices_y = [v.y for v in self.vertices]
                self.animation_vertices.append((current_vertices_x, current_vertices_y))
                
                # Сохраняем текущий путь если есть
                path = self.extract_path()
                if len(path) > 1:
                    path_x, path_y = zip(*path)
                    self.animation_paths.append((list(path_x), list(path_y)))
                else:
                    self.animation_paths.append(([], []))
                
                # Проверка достижения цели
                if self.distance(new_vertex, self.goal) < self.step_size:
                    if self.is_collision_free(new_vertex, self.goal):
                        goal_vertex = Vertex(self.goal.x, self.goal.y)
                        goal_vertex.parent = new_vertex
                        goal_vertex.cost = new_vertex.cost + self.distance(new_vertex, goal_vertex)
                        self.vertices.append(goal_vertex)
                        
                        # Сохраняем последнее ребро
                        self.animation_edges.append(
                            ([new_vertex.x, goal_vertex.x], 
                             [new_vertex.y, goal_vertex.y])
                        )
                        
                        # Финальный путь
                        final_path = self.extract_path(goal_vertex)
                        if len(final_path) > 1:
                            path_x, path_y = zip(*final_path)
                            self.animation_paths.append((list(path_x), list(path_y)))
                        
                        print(f"Цель достигнута на итерации {i}")
                        break
        
        # Создаем анимацию
        return self.create_animation(save_animation)
    
    def create_animation(self, save_animation=False):
        """Создание анимации процесса построения"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Настройка графика
        ax.set_xlim(self.area_bounds[0], self.area_bounds[2])
        ax.set_ylim(self.area_bounds[1], self.area_bounds[3])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('RRT Path Planning - Live Animation')
        
        # Отрисовка препятствий
        for obstacle in self.obstacles:
            rect = patches.Rectangle(
                (obstacle[0], obstacle[1]), 
                obstacle[2] - obstacle[0], 
                obstacle[3] - obstacle[1],
                linewidth=2, edgecolor='darkred', facecolor='red', alpha=0.7
            )
            ax.add_patch(rect)
        
        # Начальная и конечная точки
        start_point = ax.scatter([self.start.x], [self.start.y], c='green', s=150, 
                               marker='o', edgecolors='black', linewidth=2, zorder=5, label='Start')
        goal_point = ax.scatter([self.goal.x], [self.goal.y], c='red', s=150, 
                              marker='*', edgecolors='black', linewidth=2, zorder=5, label='Goal')
        
        # Элементы для анимации
        tree_scatter = ax.scatter([], [], c='blue', s=20, alpha=0.6, zorder=3, label='Tree Nodes')
        tree_lines = [ax.plot([], [], 'gray', alpha=0.4, linewidth=1, zorder=2)[0] for _ in range(len(self.animation_edges))]
        path_line = ax.plot([], [], 'orange', linewidth=3, zorder=4, label='Current Path')[0]
        
        # Текст с информацией
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.legend(loc='upper left')
        
        def animate(frame):
            # Обновляем информацию
            current_iter = min(frame, len(self.animation_vertices) - 1)
            info_text.set_text(f'Iteration: {current_iter}\nVertices: {len(self.animation_vertices[current_iter][0])}\nAlgorithm: RRT')
            
            # Обновляем вершины дерева
            if current_iter < len(self.animation_vertices):
                vertices_x, vertices_y = self.animation_vertices[current_iter]
                tree_scatter.set_offsets(np.column_stack([vertices_x, vertices_y]))
            
            # Обновляем ребра дерева
            for i, line in enumerate(tree_lines):
                if i < min(frame, len(self.animation_edges)):
                    line.set_data(self.animation_edges[i][0], self.animation_edges[i][1])
                else:
                    line.set_data([], [])
            
            # Обновляем путь
            if frame < len(self.animation_paths):
                path_x, path_y = self.animation_paths[frame]
                if len(path_x) > 1:
                    path_line.set_data(path_x, path_y)
                else:
                    path_line.set_data([], [])
            
            return [tree_scatter, path_line, info_text] + tree_lines
        
        # Создаем анимацию
        total_frames = len(self.animation_vertices)
        anim = FuncAnimation(fig, animate, frames=total_frames, 
                           interval=50, blit=True, repeat_delay=1000)
        
        if save_animation:
            print("Сохраняем анимацию...")
            anim.save('rrt_animation.gif', writer='pillow', fps=20)
            print("Анимация сохранена как 'rrt_animation.gif'")
        
        plt.show()
        return anim
    
    def real_time_animation(self):
        """Анимация в реальном времени (построение и отображение одновременно)"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Настройка графика
        ax.set_xlim(self.area_bounds[0], self.area_bounds[2])
        ax.set_ylim(self.area_bounds[1], self.area_bounds[3])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Отрисовка препятствий
        for obstacle in self.obstacles:
            rect = patches.Rectangle(
                (obstacle[0], obstacle[1]), 
                obstacle[2] - obstacle[0], 
                obstacle[3] - obstacle[1],
                linewidth=2, edgecolor='darkred', facecolor='red', alpha=0.7
            )
            ax.add_patch(rect)
        
        # Начальная и конечная точки
        ax.scatter([self.start.x], [self.start.y], c='green', s=150, 
                  marker='o', edgecolors='black', linewidth=2, zorder=5, label='Start')
        ax.scatter([self.goal.x], [self.goal.y], c='red', s=150, 
                  marker='*', edgecolors='black', linewidth=2, zorder=5, label='Goal')
        
        # Инициализация элементов отображения
        tree_lines = []
        path_line, = ax.plot([], [], 'orange', linewidth=3, zorder=4, label='Current Path')
        current_vertex_scatter = ax.scatter([], [], c='purple', s=80, zorder=4, label='Current Vertex')
        random_point_scatter = ax.scatter([], [], c='cyan', s=60, marker='x', zorder=3, label='Random Point')
        
        ax.legend(loc='upper left')
        
        # Текст с информацией
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.ion()  # Включение интерактивного режима
        plt.show()
        
        # Основной цикл планирования с визуализацией
        for i in range(self.max_iter):
            ax.set_title(f'RRT Path Planning - Iteration {i}')
            info_text.set_text(f'Iteration: {i}\nVertices: {len(self.vertices)}\nAlgorithm: RRT')
            
            # Случайная точка
            rand_point = self.generate_random_point()
            random_point_scatter.set_offsets([rand_point])
            
            # Ближайшая вершина
            nearest_vertex = self.find_nearest_vertex(rand_point)
            
            # Новая вершина
            new_vertex = self.steer(nearest_vertex, rand_point)
            
            # Обновляем отображение текущей вершины
            current_vertex_scatter.set_offsets([[new_vertex.x, new_vertex.y]])
            
            # Проверка столкновений
            if self.is_collision_free(nearest_vertex, new_vertex):
                # В базовом RRT просто добавляем вершину
                new_vertex.parent = nearest_vertex
                new_vertex.cost = nearest_vertex.cost + self.distance(nearest_vertex, new_vertex)
                self.vertices.append(new_vertex)
                
                # Отрисовка нового ребра
                if new_vertex.parent:
                    line, = ax.plot([new_vertex.parent.x, new_vertex.x], 
                                  [new_vertex.parent.y, new_vertex.y], 
                                  'blue', alpha=0.5, linewidth=1)
                    tree_lines.append(line)
                
                # Обновление пути
                path = self.extract_path()
                if len(path) > 1:
                    path_x, path_y = zip(*path)
                    path_line.set_data(path_x, path_y)
                
                # Проверка достижения цели
                if self.distance(new_vertex, self.goal) < self.step_size:
                    if self.is_collision_free(new_vertex, self.goal):
                        goal_vertex = Vertex(self.goal.x, self.goal.y)
                        goal_vertex.parent = new_vertex
                        goal_vertex.cost = new_vertex.cost + self.distance(new_vertex, goal_vertex)
                        self.vertices.append(goal_vertex)
                        
                        # Отрисовка финального ребра
                        ax.plot([new_vertex.x, goal_vertex.x], 
                              [new_vertex.y, goal_vertex.y], 
                              'blue', alpha=0.5, linewidth=1)
                        
                        # Финальный путь
                        final_path = self.extract_path(goal_vertex)
                        if len(final_path) > 1:
                            path_x, path_y = zip(*final_path)
                            path_line.set_data(path_x, path_y)
                            path_line.set_color('green')
                            path_line.set_linewidth(4)
                        
                        ax.set_title(f'RRT Path Planning - GOAL REACHED! (Iteration {i})')
                        break
            
            # Обновление отображения
            plt.draw()
            plt.pause(0.01)  # Небольшая пауза для анимации
        
        plt.ioff()  # Выключение интерактивного режима
        plt.show()
        
        return self.extract_path()

    def plan(self):
        """Основной цикл планирования без анимации"""
        start_time = time.time()
        
        for i in range(self.max_iter):
            # Случайная точка
            rand_point = self.generate_random_point()
            
            # Ближайшая вершина
            nearest_vertex = self.find_nearest_vertex(rand_point)
            
            # Новая вершина
            new_vertex = self.steer(nearest_vertex, rand_point)
            
            # Проверка столкновений
            if self.is_collision_free(nearest_vertex, new_vertex):
                # В базовом RRT просто добавляем вершину к ближайшему родителю
                new_vertex.parent = nearest_vertex
                new_vertex.cost = nearest_vertex.cost + self.distance(nearest_vertex, new_vertex)
                self.vertices.append(new_vertex)
                
                # Проверка достижения цели
                if self.distance(new_vertex, self.goal) < self.step_size:
                    if self.is_collision_free(new_vertex, self.goal):
                        goal_vertex = Vertex(self.goal.x, self.goal.y)
                        goal_vertex.parent = new_vertex
                        goal_vertex.cost = new_vertex.cost + self.distance(new_vertex, goal_vertex)
                        self.vertices.append(goal_vertex)
                        print(f"Цель достигнута на итерации {i}")
                        break
        
        end_time = time.time()
        print(f"Время выполнения: {end_time - start_time:.2f} секунд")
        return self.extract_path()

    def generate_random_point(self):
        """Генерация случайной точки (5% вероятность выбора цели)"""
        if random.random() < 0.05:
            return (self.goal.x, self.goal.y)
        
        x = random.uniform(self.area_bounds[0], self.area_bounds[2])
        y = random.uniform(self.area_bounds[1], self.area_bounds[3])
        return (x, y)

    def find_nearest_vertex(self, point):
        """Поиск ближайшей вершины к точке"""
        min_dist = float('inf')
        nearest_vertex = None
        
        for vertex in self.vertices:
            dist = self.distance(vertex, point)
            if dist < min_dist:
                min_dist = dist
                nearest_vertex = vertex
                
        return nearest_vertex

    def steer(self, from_vertex, to_point):
        """Создание новой вершины в направлении точки"""
        dist = self.distance(from_vertex, to_point)
        
        if dist <= self.step_size:
            # Если точка близко, используем ее как новую вершину
            new_vertex = Vertex(to_point[0], to_point[1])
        else:
            # Иначе двигаемся в направлении точки на step_size
            theta = math.atan2(to_point[1] - from_vertex.y, to_point[0] - from_vertex.x)
            new_x = from_vertex.x + self.step_size * math.cos(theta)
            new_y = from_vertex.y + self.step_size * math.sin(theta)
            new_vertex = Vertex(new_x, new_y)
            
        return new_vertex

    def is_collision_free(self, vertex1, vertex2):
        """Проверка отсутствия столкновений между двумя вершинами"""
        # Проверяем отрезок между vertex1 и vertex2 на пересечение с препятствиями
        for obstacle in self.obstacles:
            if self.line_rectangle_intersection(
                vertex1.x, vertex1.y, vertex2.x, vertex2.y,
                obstacle[0], obstacle[1], obstacle[2], obstacle[3]
            ):
                return False
        return True

    def line_rectangle_intersection(self, x1, y1, x2, y2, rx1, ry1, rx2, ry2):
        """Проверка пересечения отрезка и прямоугольника"""
        # Проверяем пересечение со всеми сторонами прямоугольника
        sides = [
            (rx1, ry1, rx2, ry1),  # нижняя
            (rx2, ry1, rx2, ry2),  # правая
            (rx1, ry2, rx2, ry2),  # верхняя
            (rx1, ry1, rx1, ry2)   # левая
        ]
        
        for side in sides:
            if self.line_line_intersection(x1, y1, x2, y2, side[0], side[1], side[2], side[3]):
                return True
        
        # Проверяем случай, когда отрезок полностью внутри прямоугольника
        if (rx1 <= x1 <= rx2 and ry1 <= y1 <= ry2) or (rx1 <= x2 <= rx2 and ry1 <= y2 <= ry2):
            return True
            
        return False

    def line_line_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """Проверка пересечения двух отрезков"""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        A = (x1, y1)
        B = (x2, y2)
        C = (x3, y3)
        D = (x4, y4)
        
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def extract_path(self, goal_vertex=None):
        """Извлечение финального пути"""
        if goal_vertex is None:
            # Находим вершину, ближайшую к цели
            goal_vertex = self.find_nearest_vertex((self.goal.x, self.goal.y))
        
        path = []
        current = goal_vertex
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        
        path.reverse()
        return path

    def distance(self, a, b):
        """Евклидово расстояние между двумя точками/вершинами"""
        if isinstance(a, Vertex) and isinstance(b, tuple):
            return math.sqrt((a.x - b[0])**2 + (a.y - b[1])**2)
        elif isinstance(a, tuple) and isinstance(b, Vertex):
            return math.sqrt((a[0] - b.x)**2 + (a[1] - b.y)**2)
        elif isinstance(a, Vertex) and isinstance(b, Vertex):
            return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
        else:  # оба tuple
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def visualize(self, show_tree=True, show_path=True):
        """Визуализация дерева и пути"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Отрисовка препятствий
        for obstacle in self.obstacles:
            rect = patches.Rectangle(
                (obstacle[0], obstacle[1]), 
                obstacle[2] - obstacle[0], 
                obstacle[3] - obstacle[1],
                linewidth=1, edgecolor='red', facecolor='red', alpha=0.5
            )
            ax.add_patch(rect)
        
        # Отрисовка дерева
        if show_tree:
            for vertex in self.vertices:
                if vertex.parent:
                    ax.plot([vertex.x, vertex.parent.x], [vertex.y, vertex.parent.y], 
                           'gray', alpha=0.3, linewidth=0.5)
        
        # Отрисовка вершин
        for vertex in self.vertices:
            color = 'blue' if vertex == self.start else 'red' if vertex == self.goal else 'green'
            size = 50 if vertex == self.start or vertex == self.goal else 20
            ax.scatter(vertex.x, vertex.y, c=color, s=size, zorder=3)
        
        # Отрисовка пути
        path = self.extract_path()
        if show_path and len(path) > 1:
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, 'b-', linewidth=2, zorder=2)
        
        # Настройка графика
        ax.set_xlim(self.area_bounds[0], self.area_bounds[2])
        ax.set_ylim(self.area_bounds[1], self.area_bounds[3])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('RRT Path Planning')
        
        plt.show()

# Пример использования с анимацией
if __name__ == "__main__":
    obstacles = [
        (20, 20, 40, 60),
        (60, 40, 80, 80),
        (30, 80, 70, 100)
    ]

    start = (10, 10)
    goal = (90, 90)
    area_bounds = (0, 0, 100, 100)

    # Создание планировщика RRT
    rrt = RRT(start, goal, obstacles, area_bounds, max_iter=500, step_size=5)
    
    print("Запуск RRT с анимацией...")
    
    # Вариант 1: Анимация после построения (сохраняет историю и потом анимирует)
    #path = rrt.plan_with_animation(save_animation=True)
    
    # Вариант 2: Анимация в реальном времени (раскомментируйте для использования)
    path = rrt.real_time_animation()
    
    # Вариант 3: Простое планирование без анимации
    # path = rrt.plan()
    
    print(f"Найден путь длиной {len(path)} точек")
    if path:
        print(f"Стоимость пути: {rrt.vertices[-1].cost:.2f}")
    
    # Статическая визуализация
    rrt.visualize()