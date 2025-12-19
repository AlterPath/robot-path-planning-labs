#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи
start_pos = np.array([0.0, 0.0])  # Стартовая позиция робота
goal_pos = np.array([9.0, 10.0])  # Позиция цели
obstacles = [  # Список препятствий (x, y, radius)
    np.array([3.0, 4.0, 1.0]),
    np.array([7.0, 8.0, 1.5]),
    np.array([5.0, 5.0, 1.0])
]

# Параметры потенциалов
K_att = 0.5  # Коэффициент притяжения
K_rep = 2.0  # Коэффициент отталкивания
d_safe = 2.0  # Безопасное расстояние до препятствия
max_steps = 500  # Максимальное число шагов
step_size = 0.1  # Размер шага


# Функция расчета расстояния между точками
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


# Функция расчета поля притяжения
def attractive_potential(current_pos, goal_pos):
    return 0.5 * K_att * distance(current_pos, goal_pos) ** 2


# Функция расчета силы притяжения (отрицательный график потенциала)
def attractive_force(current_pos, goal_pos):
    return K_att * (goal_pos - current_pos)


# Функция расчета поля отталкивания
def repulsive_potential(current_pos, obstacle):
    obs_pos = obstacle[:2]
    radius = obstacle[2]

    d = distance(current_pos, obs_pos)
    if d <= radius + d_safe:
        return 0.5 * K_rep * (1 / (d - radius) - 1 / d_safe) ** 2
    else:
        return 0.0


# Функция расчета силы отталкивания (отрицательный график потечного)
def repulsive_force(current_pos, obstacle):
    obs_pos = obstacle[:2]
    radius = obstacle[2]

    d = distance(current_pos, obs_pos)
    if d <= radius + d_safe:
        # Исправлено: убрана лишняя закрывающая скобка
        force = K_rep * (1 / (d - radius) - 1 / d_safe) * (1 / (d - radius) ** 2) * (current_pos - obs_pos) / d
        return force
    else:
        return np.array([0.0, 0.0])


# Основной алгоритм планирования пути
def potential_field_planning(start_pos, goal_pos, obstacles):
    path = [start_pos.copy()]
    current_pos = start_pos.copy()

    for step in range(max_steps):
        # Расчет суммарной силы
        total_force = attractive_force(current_pos, goal_pos)

        for obstacle in obstacles:
            rep_force = repulsive_force(current_pos, obstacle)
            total_force += rep_force

        # Нормализация и шаг
        force_norm = np.linalg.norm(total_force)
        if force_norm > 0:
            total_force = total_force / force_norm

        # Обновление позиции
        current_pos = current_pos + step_size * total_force
        path.append(current_pos.copy())

        # Проверка достижения цели
        if distance(current_pos, goal_pos) < 0.5:
            print(f"Цель достигнута за {step} шагов")
            break

    return np.array(path)


# Запуск планирования
path = potential_field_planning(start_pos, goal_pos, obstacles)

# Визуализация
plt.figure(figsize=(10, 8))
plt.plot(path[:, 0], path[:, 1], 'b-', label='Путь')
plt.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Старт')
plt.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Цель')

for obstacle in obstacles:
    circle = plt.Circle(obstacle[:2], obstacle[2], color='gray', alpha=0.5)
    plt.gca().add_patch(circle)
    # Зона влияния
    circle_safe = plt.Circle(obstacle[:2], obstacle[2] + d_safe, color='red', alpha=0.2, linestyle='--', fill=False)
    plt.gca().add_patch(circle_safe)

plt.grid(True)
plt.axis('equal')
plt.legend()
plt.title('Планирование пути методом потенциальных полей')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
