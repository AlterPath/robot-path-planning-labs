import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation

# ========== ФУНКЦИИ ПРИНАДЛЕЖНОСТИ ==========
dist_left = ctrl.Antecedent(np.arange(0, 6.1, 0.1), 'dist_left')
dist_front = ctrl.Antecedent(np.arange(0, 6.1, 0.1), 'dist_front')
dist_right = ctrl.Antecedent(np.arange(0, 6.1, 0.1), 'dist_right')
angle_to_goal = ctrl.Antecedent(np.arange(-180, 181, 1), 'angle_to_goal')

turn_angle = ctrl.Consequent(np.arange(-60, 61, 1), 'turn_angle')
speed = ctrl.Consequent(np.arange(0, 2.6, 0.1), 'speed')

# Функции принадлежности для расстояний
for var in [dist_left, dist_front, dist_right]:
    var['close'] = fuzz.trimf(var.universe, [0, 0, 2])
    var['medium'] = fuzz.trimf(var.universe, [1, 3, 5])
    var['far'] = fuzz.trimf(var.universe, [4, 6, 6])

# Функции принадлежности для угла до цели
angle_to_goal['right'] = fuzz.trimf(angle_to_goal.universe, [-180, -180, 0])
angle_to_goal['center'] = fuzz.trimf(angle_to_goal.universe, [-45, 0, 45])
angle_to_goal['left'] = fuzz.trimf(angle_to_goal.universe, [0, 180, 180])

# Функции принадлежности для угла поворота
turn_angle['sharp_right'] = fuzz.trimf(turn_angle.universe, [-60, -60, -30])
turn_angle['right'] = fuzz.trimf(turn_angle.universe, [-45, -15, 0])
turn_angle['straight'] = fuzz.trimf(turn_angle.universe, [-10, 0, 10])
turn_angle['left'] = fuzz.trimf(turn_angle.universe, [0, 15, 45])
turn_angle['sharp_left'] = fuzz.trimf(turn_angle.universe, [30, 60, 60])

# Функции принадлежности для скорости (изначальные)
speed['slow'] = fuzz.trimf(speed.universe, [0, 0, 0.8])
speed['medium'] = fuzz.trimf(speed.universe, [0.5, 1, 1.5])
speed['fast'] = fuzz.trimf(speed.universe, [1.2, 2.5, 2.5])

# ========== БАЗА ПРАВИЛ ==========
rule_speed_1 = ctrl.Rule(dist_front['close'], speed['slow'])
rule_speed_2 = ctrl.Rule(dist_front['medium'], speed['medium'])
rule_speed_3 = ctrl.Rule(dist_front['far'], speed['fast'])

rule_avoid_front_1 = ctrl.Rule(dist_front['close'] & dist_right['far'], turn_angle['sharp_right'])
rule_avoid_front_2 = ctrl.Rule(dist_front['close'] & dist_right['close'], turn_angle['sharp_left'])

rule_avoid_left = ctrl.Rule(dist_left['close'] | dist_left['medium'], turn_angle['right'])
rule_avoid_right = ctrl.Rule(dist_right['close'] | dist_right['medium'], turn_angle['left'])

rule_avoid_front_med = ctrl.Rule(dist_front['medium'], turn_angle['sharp_right'])

rule_goal_left = ctrl.Rule(dist_front['far'] & angle_to_goal['left'], turn_angle['sharp_left'])
rule_goal_right = ctrl.Rule(dist_front['far'] & angle_to_goal['right'], turn_angle['sharp_right'])
rule_goal_center = ctrl.Rule(dist_front['far'] & angle_to_goal['center'], turn_angle['straight'])

# ========== АДАПТИВНАЯ СИСТЕМА ==========
class AdaptiveFuzzySystem:
    def __init__(self):
        self.base_speed = 1.0  # Базовая скорость для адаптации
        self.adaptation_factor = 1.0
        self.obstacle_count = 0
        
    def update_adaptation(self, distances, angle_diff):
        """Адаптация параметров системы в реальном времени"""
        # Подсчет близких препятствий
        close_obstacles = sum(1 for d in distances if d < 2)
        self.obstacle_count = close_obstacles
        
        # Адаптация на основе количества препятствий
        if close_obstacles >= 2:
            self.adaptation_factor = 0.6  # Сильное замедление
        elif close_obstacles == 1:
            self.adaptation_factor = 0.8  # Умеренное замедление
        else:
            self.adaptation_factor = 1.0  # Нормальная скорость
            
        # Дополнительная адаптация по углу до цели
        if abs(angle_diff) > 90:
            self.adaptation_factor *= 0.9  # Замедление при резких поворотах
            
        return self.adaptation_factor

# ========== ГРАФИКИ ТЕРМОВ ==========
def plot_membership_functions():
    """Визуализация функций принадлежности"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Графики для входных переменных
    dist_front.view(ax=axes[0, 0])
    axes[0, 0].set_title('Расстояние спереди')
    
    dist_left.view(ax=axes[0, 1])
    axes[0, 1].set_title('Расстояние слева')
    
    dist_right.view(ax=axes[0, 2])
    axes[0, 2].set_title('Расстояние справа')
    
    angle_to_goal.view(ax=axes[1, 0])
    axes[1, 0].set_title('Угол до цели')
    
    # Графики для выходных переменных
    turn_angle.view(ax=axes[1, 1])
    axes[1, 1].set_title('Угол поворота')
    
    speed.view(ax=axes[1, 2])
    axes[1, 2].set_title('Скорость')
    
    plt.tight_layout()
    plt.savefig('membership_functions.png', dpi=150)
    plt.show()

# Выводим графики термов
plot_membership_functions()

# ========== СИСТЕМА УПРАВЛЕНИЯ ==========
robot_ctrl = ctrl.ControlSystem([
    rule_speed_1, rule_speed_2, rule_speed_3,
    rule_avoid_front_1, rule_avoid_front_2,
    rule_avoid_left, rule_avoid_right, rule_avoid_front_med,
    rule_goal_left, rule_goal_right, rule_goal_center
])

robot_sim = ctrl.ControlSystemSimulation(robot_ctrl)
adaptive_system = AdaptiveFuzzySystem()

# ========== КЛАССЫ ДЛЯ СИМУЛЯЦИИ ==========
class Obstacle:
    def __init__(self, x, y, r, vx=0.0, vy=0.0, omega=0.0, movable=False):
        self.x = x
        self.y = y
        self.r = r
        self.vx = vx
        self.vy = vy
        self.omega = omega
        self.movable = movable
        self.time = 0.0
        self.original_x = x
        self.original_y = y
        
    def update(self, dt):
        if self.movable:
            self.time += dt
            if self.omega != 0:
                self.x = self.original_x + np.sin(self.time * self.omega) * 1.0
            else:
                self.x += self.vx * dt
                self.y += self.vy * dt

class AnimatedRobotSim:
    def __init__(self, start, goal, obstacles):
        self.pos = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.obstacles = obstacles
        self.heading = 45.0
        self.path = [self.pos.copy()]
        self.dt = 0.1
        self.reached = False
        self.collisions = 0
        self.steps = 0
        
    def sense(self):
        to_goal = self.goal - self.pos
        goal_angle = np.degrees(np.arctan2(to_goal[1], to_goal[0]))
        angle_diff = goal_angle - self.heading
        angle_diff = (angle_diff + 180) % 360 - 180
        
        directions = [self.heading + 90, self.heading, self.heading - 90]
        distances = []
        
        for d in directions:
            rad = np.radians(d)
            ray = np.array([np.cos(rad), np.sin(rad)])
            min_dist = 6.0
            
            for obs in self.obstacles:
                to_obs = np.array([obs.x, obs.y]) - self.pos
                proj = np.dot(to_obs, ray)
                
                if proj > 0:
                    closest = self.pos + proj * ray
                    dist_to_obs = np.linalg.norm(closest - np.array([obs.x, obs.y]))
                    
                    if dist_to_obs < obs.r:
                        min_dist = min(min_dist, proj)
                        if proj < obs.r + 0.1:  # Регистрация столкновения
                            self.collisions += 1
            
            distances.append(min(min_dist, 6.0))
        
        return distances[0], distances[1], distances[2], angle_diff
    
    def step(self):
        if self.reached:
            return
            
        for obs in self.obstacles:
            obs.update(self.dt)
        
        d_left, d_front, d_right, ang = self.sense()
        
        # Адаптация системы
        adaptation = adaptive_system.update_adaptation([d_left, d_front, d_right], ang)
        
        robot_sim.input['dist_left'] = d_left
        robot_sim.input['dist_front'] = d_front
        robot_sim.input['dist_right'] = d_right
        robot_sim.input['angle_to_goal'] = ang
        
        try:
            robot_sim.compute()
            turn = robot_sim.output['turn_angle']
            v = robot_sim.output['speed']
        except Exception as e:
            turn, v = 0, 0
        
        # Применяем адаптацию к скорости
        v *= adaptation
        
        self.heading += turn
        self.heading = (self.heading + 180) % 360 - 180
        
        dx = v * np.cos(np.radians(self.heading)) * self.dt
        dy = v * np.sin(np.radians(self.heading)) * self.dt
        
        self.pos += np.array([dx, dy])
        self.path.append(self.pos.copy())
        self.steps += 1
        
        if np.linalg.norm(self.pos - self.goal) < 0.5:
            self.reached = True

# ========== ПАРАМЕТРЫ СИМУЛЯЦИИ ==========
start = (1, 1)
goal = (9, 8)

obstacles = [
    Obstacle(5, 5, 0.8),
    Obstacle(3, 7, 0.6),
    Obstacle(7, 3, 0.5, vx=0.0, vy=0.0, omega=0.8, movable=True),
    Obstacle(2, 2, 0.4, vx=0.15, vy=0.0, movable=True),
    Obstacle(8, 6, 0.5, vx=0.0, vy=-0.12, movable=True),
]

sim = AnimatedRobotSim(start, goal, obstacles)

# ========== ВИЗУАЛИЗАЦИЯ ==========
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("Robot Navigation with Fuzzy Logic (Adaptive System)")
ax.grid(True)
ax.set_aspect('equal')

robot_point, = ax.plot([], [], 'bo', markersize=12, label='Robot')
path_line, = ax.plot([], [], 'b-', alpha=0.7, linewidth=2)
ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
ax.plot(start[0], start[1], 'go', markersize=10, label='Start')

# Текст для отображения адаптации
adapt_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

obs_patches = []
for obs in obstacles:
    color = 'red' if obs.movable else 'gray'
    label_text = 'Dynamic' if obs.movable else 'Static'
    patch = patches.Circle((obs.x, obs.y), obs.r, color=color, alpha=0.6, label=label_text)
    ax.add_patch(patch)
    obs_patches.append(patch)

ax.legend()

def init():
    robot_point.set_data([], [])
    path_line.set_data([], [])
    adapt_text.set_text('')
    return [robot_point, path_line, adapt_text] + obs_patches

def animate(frame):
    sim.step()
    
    robot_point.set_data([sim.pos[0]], [sim.pos[1]])
    
    path = np.array(sim.path)
    path_line.set_data(path[:, 0], path[:, 1])
    
    for patch, obs in zip(obs_patches, obstacles):
        patch.center = (obs.x, obs.y)
    
    # Обновление информации об адаптации
    adapt_info = (f"Adaptation: {adaptive_system.adaptation_factor:.2f}\n"
                  f"Close obstacles: {adaptive_system.obstacle_count}\n"
                  f"Steps: {sim.steps}")
    adapt_text.set_text(adapt_info)
    
    return [robot_point, path_line, adapt_text] + obs_patches

ani = FuncAnimation(fig, animate, init_func=init,
                    frames=500, interval=50, blit=True, repeat=False)

plt.show()

# ========== АНАЛИЗ РЕЗУЛЬТАТОВ ==========
print("="*50)
print("РЕЗУЛЬТАТЫ СИМУЛЯЦИИ")
print("="*50)
print(f"Цель достигнута: {'ДА' if sim.reached else 'НЕТ'}")
print(f"Шагов выполнено: {sim.steps}")
print(f"Столкновений: {sim.collisions}")
print(f"Длина пути: {len(sim.path)} точек")
if sim.reached:
    final_dist = np.linalg.norm(sim.pos - sim.goal)
    print(f"Финальное расстояние до цели: {final_dist:.2f}")
    
print(f"\nПАРАМЕТРЫ АДАПТАЦИИ:")
print(f"Фактор адаптации: {adaptive_system.adaptation_factor:.2f}")
print(f"Максимальная скорость: {adaptive_system.base_speed * adaptive_system.adaptation_factor:.2f}")

# График пути
plt.figure(figsize=(8, 8))
plt.plot([p[0] for p in sim.path], [p[1] for p in sim.path], 'b-', alpha=0.7, label='Путь')
plt.plot(start[0], start[1], 'go', markersize=10, label='Старт')
plt.plot(goal[0], goal[1], 'r*', markersize=15, label='Цель')

for obs in obstacles:
    color = 'red' if obs.movable else 'gray'
    circle = plt.Circle((obs.x, obs.y), obs.r, color=color, alpha=0.5)
    plt.gca().add_patch(circle)

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.grid(True)
plt.legend()
plt.title("Траектория движения робота")
plt.savefig('robot_path.png', dpi=150)
plt.show()