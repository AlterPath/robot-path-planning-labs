import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
from typing import List, Tuple, Optional


class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0  # стоимость пути от старта
        self.parent = None


class RRTStar:
    def __init__(self, start, goal, obstacles, area_bounds, max_iter=1000, step_size=5, neighbor_radius=15):
        self.start = Vertex(start[0], start[1])
        self.goal = Vertex(goal[0], goal[1])
        self.obstacles = obstacles  # список прямоугольников [(x1,y1,x2,y2)]
        self.area_bounds = area_bounds  # (x_min, y_min, x_max, y_max)
        self.max_iter = max_iter
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.vertices = [self.start]
        self.goal_reached = False
        self.goal_vertex = None
        self.iteration_history = []  # Для хранения истории итераций для анимации

    def plan(self):
        """Основной цикл планирования RRT*"""
        for iteration in range(self.max_iter):
            # Сохраняем текущее состояние для анимации
            self.iteration_history.append(self.vertices.copy())

            # Генерация случайной точки
            random_point = self.generate_random_point()

            # Поиск ближайшей вершины
            nearest_vertex = self.find_nearest_vertex(random_point)

            # Создание новой вершины
            new_vertex = self.steer(nearest_vertex, random_point)

            # Проверка столкновений
            if self.is_collision_free(nearest_vertex, new_vertex):
                # Поиск вершин в окрестности
                near_vertices = self.find_near_vertices(new_vertex)

                # Выбор оптимального родителя
                self.choose_parent(new_vertex, near_vertices)

                # Добавление новой вершины в дерево
                self.vertices.append(new_vertex)

                # Переподключение вершин
                self.rewire(new_vertex, near_vertices)

                # Проверка достижения цели
                if not self.goal_reached and self.is_goal_reached(new_vertex):
                    print(f"Цель достигнута на итерации {iteration}!")
                    self.goal_reached = True
                    self.goal_vertex = new_vertex

                    # Добавляем саму цель как финальную вершину
                    if self.is_collision_free(new_vertex, self.goal):
                        goal_vertex = Vertex(self.goal.x, self.goal.y)
                        goal_vertex.parent = new_vertex
                        goal_vertex.cost = new_vertex.cost + math.sqrt(
                            (goal_vertex.x - new_vertex.x) ** 2 + (goal_vertex.y - new_vertex.y) ** 2
                        )
                        self.vertices.append(goal_vertex)
                        self.goal_vertex = goal_vertex

                    # Сохраняем финальное состояние
                    self.iteration_history.append(self.vertices.copy())
                    return self.extract_path()

        # Сохраняем финальное состояние
        self.iteration_history.append(self.vertices.copy())
        print("Достигнуто максимальное количество итераций")
        if self.goal_reached:
            return self.extract_path()
        return None

    def generate_random_point(self):
        """Генерация случайной точки"""
        x_min, y_min, x_max, y_max = self.area_bounds
        if random.random() < 0.1:  # Уменьшил вероятность выбора цели для лучшего исследования
            return (self.goal.x, self.goal.y)
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        return (x, y)

    def find_nearest_vertex(self, point):
        """Поиск ближайшей вершины к точке"""
        min_dist = float('inf')
        nearest_vertex = None

        for vertex in self.vertices:
            dist = math.sqrt((vertex.x - point[0]) ** 2 + (vertex.y - point[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_vertex = vertex

        return nearest_vertex

    def steer(self, from_vertex, to_point):
        """Создание новой вершины в направлении точки"""
        dx = to_point[0] - from_vertex.x
        dy = to_point[1] - from_vertex.y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance < 1e-6:
            return Vertex(from_vertex.x, from_vertex.y)

        dx /= distance
        dy /= distance

        actual_step = min(self.step_size, distance)

        new_x = from_vertex.x + dx * actual_step
        new_y = from_vertex.y + dy * actual_step

        x_min, y_min, x_max, y_max = self.area_bounds
        new_x = max(x_min, min(new_x, x_max))
        new_y = max(y_min, min(new_y, y_max))

        new_vertex = Vertex(new_x, new_y)
        return new_vertex

    def is_collision_free(self, vertex1, vertex2):
        """Проверка отсутствия столкновений между двумя вершинами"""
        # Сначала проверяем, не находятся ли вершины внутри препятствий
        for obs in self.obstacles:
            if self._point_in_obstacle(vertex1.x, vertex1.y, obs) or \
                    self._point_in_obstacle(vertex2.x, vertex2.y, obs):
                return False

        # Проверяем пересечение отрезка со всеми сторонами всех препятствий
        for obs in self.obstacles:
            if self._line_intersects_obstacle(vertex1, vertex2, obs):
                return False

        return True

    def _point_in_obstacle(self, x, y, obstacle):
        """Проверка, находится ли точка внутри препятствия"""
        x1, y1, x2, y2 = obstacle
        return x1 <= x <= x2 and y1 <= y <= y2

    def _line_intersects_obstacle(self, v1, v2, obstacle):
        """Проверка пересечения отрезка с прямоугольным препятствием"""
        x1, y1, x2, y2 = obstacle

        # Получаем 4 стороны прямоугольника
        sides = [
            ((x1, y1), (x2, y1)),  # нижняя
            ((x2, y1), (x2, y2)),  # правая
            ((x2, y2), (x1, y2)),  # верхняя
            ((x1, y2), (x1, y1))  # левая
        ]

        # Проверяем пересечение с каждой стороной
        for side_start, side_end in sides:
            if self._line_segments_intersect(
                    (v1.x, v1.y), (v2.x, v2.y),
                    side_start, side_end
            ):
                return True

        return False

    def _line_segments_intersect(self, a, b, c, d):
        """Проверка пересечения двух отрезков AB и CD"""

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    def find_near_vertices(self, vertex):
        """Поиск вершин в окрестности новой вершины"""
        near_vertices = []
        radius = self.neighbor_radius

        # Динамическое изменение радиуса в зависимости от размера дерева
        if len(self.vertices) > 10:
            radius = min(self.neighbor_radius * 2,
                         self.neighbor_radius * math.log(len(self.vertices)) * 0.5)

        for v in self.vertices:
            dist = math.sqrt((v.x - vertex.x) ** 2 + (v.y - vertex.y) ** 2)
            if dist <= radius and v != vertex:
                near_vertices.append(v)

        return near_vertices

    def choose_parent(self, new_vertex, near_vertices):
        """Выбор оптимального родителя среди соседних вершин"""
        min_cost = float('inf')
        best_parent = None

        for vertex in near_vertices:
            # Проверяем, свободен ли путь
            if self.is_collision_free(vertex, new_vertex):
                # Вычисляем стоимость через эту вершину
                cost = vertex.cost + math.sqrt(
                    (new_vertex.x - vertex.x) ** 2 + (new_vertex.y - vertex.y) ** 2
                )

                if cost < min_cost:
                    min_cost = cost
                    best_parent = vertex

        # Устанавливаем родителя
        if best_parent is not None:
            new_vertex.parent = best_parent
            new_vertex.cost = min_cost
        else:
            # Если родитель не найден среди соседей, используем ближайшую вершину
            new_vertex.cost = math.sqrt(
                (new_vertex.x - self.start.x) ** 2 + (new_vertex.y - self.start.y) ** 2
            )

    def rewire(self, new_vertex, near_vertices):
        """Переподключение вершин для улучшения путей"""
        for vertex in near_vertices:
            # Пропускаем, если vertex является родителем new_vertex
            if new_vertex.parent and vertex == new_vertex.parent:
                continue

            # Проверяем, можно ли улучшить путь к vertex через new_vertex
            if self.is_collision_free(new_vertex, vertex):
                new_cost = new_vertex.cost + math.sqrt(
                    (vertex.x - new_vertex.x) ** 2 + (vertex.y - new_vertex.y) ** 2
                )

                if new_cost < vertex.cost:
                    vertex.parent = new_vertex
                    vertex.cost = new_cost

                    # Рекурсивно обновляем стоимость дочерних вершин
                    self._update_children_cost(vertex)

    def _update_children_cost(self, vertex):
        """Рекурсивное обновление стоимости дочерних вершин"""
        for v in self.vertices:
            if v.parent == vertex:
                v.cost = vertex.cost + math.sqrt(
                    (v.x - vertex.x) ** 2 + (v.y - vertex.y) ** 2
                )
                self._update_children_cost(v)

    def is_goal_reached(self, vertex):
        """Проверка достижения цели"""
        dist = math.sqrt((vertex.x - self.goal.x) ** 2 + (vertex.y - self.goal.y) ** 2)
        return dist <= self.step_size * 1.5

    def extract_path(self, goal_vertex=None):
        """Извлечение финального пути"""
        if goal_vertex is None:
            goal_vertex = self.goal_vertex

        if goal_vertex is None:
            return None

        path = []
        current = goal_vertex
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent

        path.reverse()
        return path

    def calculate_path_length(self, path):
        """Вычисляет длину пути"""
        if path is None or len(path) < 2:
            return 0.0

        length = 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            length += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return length

    def create_animation(self):
        """Создает анимацию процесса построения дерева"""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Настройка графика
        ax.set_xlim(self.area_bounds[0], self.area_bounds[2])
        ax.set_ylim(self.area_bounds[1], self.area_bounds[3])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Отображение препятствий
        for obs in self.obstacles:
            x1, y1, x2, y2 = obs
            ax.fill([x1, x2, x2, x1], [y1, y1, y2, y2], 'lightblue', alpha=0.7)

        # Начальная и конечная точки
        start_point, = ax.plot(self.start.x, self.start.y, 'go', markersize=12,
                               label='Start', markeredgecolor='black', linewidth=2)
        goal_point, = ax.plot(self.goal.x, self.goal.y, 'ro', markersize=12,
                              label='Goal', markeredgecolor='black', linewidth=2)

        # Элементы для анимации
        tree_lines = []
        vertex_points = []

        def animate(frame):
            # Очищаем предыдущие элементы
            for line in tree_lines:
                line.remove()
            for point in vertex_points:
                point.remove()
            tree_lines.clear()
            vertex_points.clear()

            # Получаем состояние на текущем кадре
            vertices = self.iteration_history[frame]

            # Рисуем только валидные соединения (без столкновений)
            for vertex in vertices:
                if vertex.parent:
                    # Проверяем, свободен ли путь перед отрисовкой
                    if self.is_collision_free(vertex.parent, vertex):
                        line, = ax.plot([vertex.parent.x, vertex.x], [vertex.parent.y, vertex.y],
                                        'blue', alpha=0.4, linewidth=1.5)
                        tree_lines.append(line)

                point, = ax.plot(vertex.x, vertex.y, 'bo', markersize=2, alpha=0.6)
                vertex_points.append(point)

            # Обновляем заголовок
            if self.goal_reached and frame == len(self.iteration_history) - 1:
                ax.set_title("RRT* - Дерево построено (цель достигнута)", fontsize=14, fontweight='bold')
            elif frame == len(self.iteration_history) - 1:
                ax.set_title("RRT* - Дерево построено (цель не достигнута)", fontsize=14, fontweight='bold')
            else:
                ax.set_title(f"RRT* - Построение дерева (итерация {frame})", fontsize=14, fontweight='bold')

            return tree_lines + vertex_points

        # Создаем анимацию с одним проходом
        anim = animation.FuncAnimation(
            fig, animate, frames=len(self.iteration_history),
            interval=100, blit=True, repeat=False
        )

        ax.legend()
        plt.tight_layout()
        plt.show()

        return anim

    def visualize_final_path(self, path=None, title="Финальный путь RRT*"):
        """Визуализация финального пути"""
        if path is None:
            path = self.extract_path()

        plt.figure(figsize=(12, 10))

        # Отображение препятствий
        for obs in self.obstacles:
            x1, y1, x2, y2 = obs
            plt.fill([x1, x2, x2, x1], [y1, y1, y2, y2], 'lightblue', alpha=0.7,
                     label='Obstacles' if obs == self.obstacles[0] else "")

        # Отображение начальной и конечной точек
        plt.plot(self.start.x, self.start.y, 'go', markersize=12, label='Start',
                 markeredgecolor='black', linewidth=2)
        plt.plot(self.goal.x, self.goal.y, 'ro', markersize=12, label='Goal',
                 markeredgecolor='black', linewidth=2)

        # Отображение только валидных соединений дерева
        for vertex in self.vertices:
            if vertex.parent and self.is_collision_free(vertex.parent, vertex):
                plt.plot([vertex.parent.x, vertex.x], [vertex.parent.y, vertex.y],
                         'blue', alpha=0.2, linewidth=1)

        # Отображение всех вершин дерева
        for vertex in self.vertices:
            plt.plot(vertex.x, vertex.y, 'bo', markersize=1, alpha=0.3)

        # Отображение пути
        if path and len(path) > 1:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_x, path_y, 'red', linewidth=4, label='Final Path')
            plt.plot(path_x, path_y, 'ro', markersize=5, alpha=0.8)

            path_length = self.calculate_path_length(path)
            title += f" (Длина пути: {path_length:.2f})"

        plt.xlim(self.area_bounds[0], self.area_bounds[2])
        plt.ylim(self.area_bounds[1], self.area_bounds[3])
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal')
        plt.show()


def main():
    """Основная функция для демонстрации RRT*"""
    # Определение препятствий
    obstacles = [
        (20, 20, 40, 60),
        (60, 40, 80, 80),
        (30, 80, 70, 100)
    ]

    # Параметры алгоритма
    start = (10, 10)
    goal = (90, 90)
    area_bounds = (0, 0, 100, 100)

    # Создание и запуск планировщика RRT*
    rrt_star = RRTStar(start, goal, obstacles, area_bounds, max_iter=1000, step_size=5, neighbor_radius=15)

    import time
    start_time = time.time()
    path = rrt_star.plan()
    end_time = time.time()

    execution_time = end_time - start_time

    # Финальная визуализация
    if path:
        path_length = rrt_star.calculate_path_length(path)
        print(f"Найден путь длиной {len(path)} точек")
        print(f"Длина пути: {path_length:.2f} единиц")
        print(f"Время выполнения: {execution_time:.2f} секунд")
        print(f"Количество вершин в дереве: {len(rrt_star.vertices)}")

        # Показываем анимацию построения дерева
        print("Запуск анимации построения дерева...")
        rrt_star.create_animation()

        # Показываем отдельную картинку с финальным путем
        print("Показ финального пути...")
        rrt_star.visualize_final_path(path)

    else:
        print("Путь не найден")
        print(f"Время выполнения: {execution_time:.2f} секунд")
        print(f"Количество вершин в дереве: {len(rrt_star.vertices)}")
        # Показываем анимацию построения дерева
        print("Запуск анимации построения дерева...")
        rrt_star.create_animation()


if __name__ == "__main__":
    main()

