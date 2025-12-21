import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
import time
from typing import List, Tuple, Optional


class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0  # стоимость пути от старта
        self.parent = None


class RRT:
    def __init__(self, start, goal, obstacles, area_bounds, max_iter=1000, step_size=10):
        self.start = Vertex(start[0], start[1])
        self.goal = Vertex(goal[0], goal[1])
        self.obstacles = obstacles  # список прямоугольников [(x1,y1,x2,y2)]
        self.area_bounds = area_bounds  # (x_min, y_min, x_max, y_max)
        self.max_iter = max_iter
        self.step_size = step_size
        self.vertices = [self.start]
        self.goal_reached = False
        self.goal_vertex = None
        self.iteration_history = []  # Для хранения истории итераций для анимации
        self.planning_time = 0.0  # Время выполнения планирования

    def plan(self):
        """Основной цикл планирования"""
        start_time = time.time()

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
                # Добавление новой вершины
                new_vertex.parent = nearest_vertex
                new_vertex.cost = nearest_vertex.cost + math.sqrt(
                    (new_vertex.x - nearest_vertex.x) ** 2 + (new_vertex.y - nearest_vertex.y) ** 2
                )
                self.vertices.append(new_vertex)

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

                    # Замер времени выполнения
                    self.planning_time = time.time() - start_time
                    return self.extract_path()

        # Сохраняем финальное состояние
        self.iteration_history.append(self.vertices.copy())

        # Замер времени выполнения
        self.planning_time = time.time() - start_time

        print("Достигнуто максимальное количество итераций")
        if self.goal_reached:
            return self.extract_path()
        return None

    def get_planning_time(self):
        """Возвращает время выполнения планирования"""
        return self.planning_time

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

    def generate_random_point(self):
        """Генерация случайной точки"""
        x_min, y_min, x_max, y_max = self.area_bounds
        if random.random() < 0.15:
            return (self.goal.x, self.goal.y)
        x = random.randint(x_min, x_max - 1)
        y = random.randint(y_min, y_max - 1)
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

        if distance == 0:
            return Vertex(from_vertex.x, from_vertex.y)

        dx /= distance
        dy /= distance

        actual_step = min(self.step_size, distance)

        new_x = int(from_vertex.x + dx * actual_step)
        new_y = int(from_vertex.y + dy * actual_step)

        x_min, y_min, x_max, y_max = self.area_bounds
        new_x = max(x_min, min(new_x, x_max - 1))
        new_y = max(y_min, min(new_y, y_max - 1))

        new_vertex = Vertex(new_x, new_y)
        return new_vertex

    def is_collision_free(self, vertex1, vertex2):
        """Проверка отсутствия столкновений между двумя вершинами"""
        for obs in self.obstacles:
            x1, y1, x2, y2 = obs
            if (x1 <= vertex1.x < x2 and y1 <= vertex1.y < y2 and
                    x1 <= vertex2.x < x2 and y1 <= vertex2.y < y2):
                return True

        points = self._get_line_points((vertex1.x, vertex1.y), (vertex2.x, vertex2.y))

        for x, y in points:
            if (x < self.area_bounds[0] or x >= self.area_bounds[2] or
                    y < self.area_bounds[1] or y >= self.area_bounds[3]):
                return False

            for obs in self.obstacles:
                x1, y1, x2, y2 = obs
                if x1 <= x < x2 and y1 <= y < y2:
                    return False

        return True

    def _get_line_points(self, point1, point2):
        """Возвращает все точки на линии между point1 и point2"""
        x1, y1 = point1
        x2, y2 = point2
        points = []

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        x, y = x1, y1
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1

        error = dx - dy

        max_steps = max(dx, dy) * 3
        steps = 0

        while (x != x2 or y != y2) and steps < max_steps:
            points.append((x, y))

            error2 = error * 2

            if error2 > -dy:
                error -= dy
                x += x_inc

            if error2 < dx:
                error += dx
                y += y_inc

            steps += 1

        points.append((x2, y2))
        return points

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

    def create_animation(self):
        """Создает анимацию процесса построения дерева (без финального пути)"""
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

            # Рисуем дерево
            for vertex in vertices:
                if vertex.parent:
                    line, = ax.plot([vertex.parent.x, vertex.x], [vertex.parent.y, vertex.y],
                                    'blue', alpha=0.4, linewidth=1.5)
                    tree_lines.append(line)

                point, = ax.plot(vertex.x, vertex.y, 'bo', markersize=2, alpha=0.6)
                vertex_points.append(point)

            # Обновляем заголовок с информацией о времени
            time_info = f" | Время планирования: {self.planning_time:.3f} с"

            if self.goal_reached and frame == len(self.iteration_history) - 1:
                ax.set_title("RRT - Дерево построено (цель достигнута)" + time_info, fontsize=14, fontweight='bold')
            elif frame == len(self.iteration_history) - 1:
                ax.set_title("RRT - Дерево построено (цель не достигнута)" + time_info, fontsize=14, fontweight='bold')
            else:
                ax.set_title(f"RRT - Построение дерева (итерация {frame})", fontsize=14, fontweight='bold')

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

    def visualize_final_path(self, path=None, title="Финальный путь RRT"):
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

        # Отображение всего дерева (более прозрачное)
        for vertex in self.vertices:
            if vertex.parent:
                plt.plot([vertex.parent.x, vertex.x], [vertex.parent.y, vertex.y],
                         'blue', alpha=0.2, linewidth=1)

        # Отображение всех вершин дерева (более прозрачные)
        for vertex in self.vertices:
            plt.plot(vertex.x, vertex.y, 'bo', markersize=1, alpha=0.3)

        # Отображение пути (яркий красный)
        if path and len(path) > 1:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_x, path_y, 'red', linewidth=4, label='Final Path')
            plt.plot(path_x, path_y, 'ro', markersize=5, alpha=0.8)

            path_length = self.calculate_path_length(path)
            title += f" (Длина пути: {path_length:.2f} ед.)"

        # Добавляем информацию о времени в заголовок
        title += f" | Время: {self.planning_time:.3f} с"
        title += f" | Вершин: {len(self.vertices)}"

        plt.xlim(self.area_bounds[0], self.area_bounds[2])
        plt.ylim(self.area_bounds[1], self.area_bounds[3])
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal')
        plt.show()


# Пример использования
def main():
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

    # Создание и запуск планировщика
    rrt = RRT(start, goal, obstacles, area_bounds, max_iter=500, step_size=8)

    print("Запуск планирования пути...")
    path = rrt.plan()

    # Получаем время выполнения
    planning_time = rrt.get_planning_time()

    # Финальная визуализация
    if path:
        path_length = rrt.calculate_path_length(path)
        print(f"Найден путь длиной {len(path)} точек")
        print(f"Длина пути: {path_length:.2f} единиц")
        print(f"Время выполнения: {planning_time:.3f} секунд")
        print(f"Количество вершин в дереве: {len(rrt.vertices)}")
        print(f"Скорость планирования: {len(rrt.vertices) / planning_time:.1f} вершин/сек")

        # Показываем анимацию построения дерева
        print("\nЗапуск анимации построения дерева...")
        rrt.create_animation()

        # Показываем отдельную картинку с финальным путем
        print("\nПоказ финального пути...")
        rrt.visualize_final_path(path)

    else:
        print("Путь не найден")
        print(f"Время выполнения: {planning_time:.3f} секунд")
        print(f"Количество вершин в дереве: {len(rrt.vertices)}")
        # Показываем анимацию построения дерева
        print("\nЗапуск анимации построения дерева...")
        rrt.create_animation()


if __name__ == "__main__":
    main()