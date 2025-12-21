import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import time

pio.renderers.default = "browser"

GLOBAL_MAXIMA = np.array([
    [3.0, 2.0],
    [-2.805118, 3.131312],
    [-3.779310, -3.283186],
    [3.584428, -1.848126]
])

def himmelblau(x, y):
    return -((x**2 + y - 11)**2 + (x + y**2 - 7)**2)

def create_population(size, bounds=(-5, 5)):
    low, high = bounds
    pop = np.random.uniform(low, high, (size, 2))
    return pop

def evaluate(population):
    return np.array([himmelblau(x, y) for x, y in population])

def select_parents(population, fitness, num_parents):
    parents = []
    for _ in range(num_parents):
        idx = np.random.choice(len(population), 3, replace=False)
        best_idx = idx[np.argmax(fitness[idx])]
        parents.append(population[best_idx])
    return np.array(parents)

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.random.randint(1, offspring_size[1])
    
    for k in range(offspring_size[0]):
        parent1_idx = k % len(parents)
        parent2_idx = (k + 1) % len(parents)
        
        offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    
    return offspring

def mutate(offspring, mutation_rate=0.1, bounds=(-5, 5)):
    for idx in range(len(offspring)):
        if np.random.random() < mutation_rate:
            offspring[idx] += np.random.normal(0, 0.5, 2)
            offspring[idx] = np.clip(offspring[idx], bounds[0], bounds[1])
    return offspring

def genetic_algorithm(pop_size=50, generations=30, bounds=(-5, 5)):
    population = create_population(pop_size, bounds)
    best_fitness_history = []
    avg_fitness_history = []
    all_populations = []
    best_individual = None
    best_fitness = -np.inf
    
    for generation in range(generations):
        all_populations.append(population.copy())
        fitness = evaluate(population)
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_fitness:
            best_fitness = fitness[current_best_idx]
            best_individual = population[current_best_idx].copy()
        
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(np.mean(fitness))
        num_parents = max(2, int(pop_size * 0.3))
        parent_indices = np.argsort(fitness)[-num_parents:]
        parents = population[parent_indices]
        offspring = crossover(parents, (pop_size - len(parents), 2))
        offspring = mutate(offspring, mutation_rate=0.1, bounds=bounds)
        population[:len(parents)] = parents
        population[len(parents):] = offspring
        
        if generation % 5 == 0:
            print(f"Поколение {generation+1:3d}: Лучшее значение = {best_fitness:8.4f}")
    
    return (best_individual, best_fitness, best_fitness_history, 
            avg_fitness_history, all_populations)

def create_3d_surface(bounds=(-5, 5)):
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = himmelblau(X, Y)
    return X, Y, Z

def create_contour_plot(bounds=(-5, 5)):
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = himmelblau(X, Y)
    return X, Y, Z

def plot_interactive_results(best_point, history_best, history_avg, all_populations, bounds=(-5, 5)):
    X_surf, Y_surf, Z_surf = create_3d_surface(bounds)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('3D Поверхность функции', 'Контурный график с популяцией',
                       'Сходимость алгоритма', 'Эволюция популяции'),
        specs=[[{'type': 'surface'}, {'type': 'contour'}],
               [{'type': 'xy'}, {'type': 'xy'}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    fig.add_trace(
        go.Surface(
            x=X_surf, y=Y_surf, z=Z_surf,
            colorscale='Viridis',
            opacity=0.8,
            showscale=True,
            contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
        ),
        row=1, col=1
    )
    
    for point in GLOBAL_MAXIMA:
        x, y = point
        z = himmelblau(x, y)
        fig.add_trace(
            go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(size=10, color='red', symbol='diamond'),
                name='Глобальный максимум',
                showlegend=True if (point == GLOBAL_MAXIMA[0]).all() else False
            ),
            row=1, col=1
        )
    
    z_best = himmelblau(best_point[0], best_point[1])
    fig.add_trace(
        go.Scatter3d(
            x=[best_point[0]], y=[best_point[1]], z=[z_best],
            mode='markers',
            marker=dict(size=12, color='white', symbol='circle',
                       line=dict(color='red', width=3)),
            name='Найденный максимум'
        ),
        row=1, col=1
    )
    
    X_cont, Y_cont, Z_cont = create_contour_plot(bounds)
    
    fig.add_trace(
        go.Contour(
            x=X_cont[0], y=Y_cont[:, 0], z=Z_cont,
            colorscale='Viridis',
            showscale=False,
            contours=dict(showlabels=True)
        ),
        row=1, col=2
    )
    
    for point in GLOBAL_MAXIMA:
        fig.add_trace(
            go.Scatter(
                x=[point[0]], y=[point[1]],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='Глобальный максимум',
                showlegend=False
            ),
            row=1, col=2
        )
    
    final_pop = all_populations[-1]
    fig.add_trace(
        go.Scatter(
            x=final_pop[:, 0], y=final_pop[:, 1],
            mode='markers',
            marker=dict(size=8, color='cyan', symbol='circle',
                       line=dict(color='black', width=1)),
            name='Финальная популяция'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=[best_point[0]], y=[best_point[1]],
            mode='markers',
            marker=dict(size=15, color='white', symbol='circle',
                       line=dict(color='red', width=3)),
            name='Найденный максимум',
            showlegend=False
        ),
        row=1, col=2
    )
    
    generations = list(range(len(history_best)))
    
    fig.add_trace(
        go.Scatter(
            x=generations, y=history_best,
            mode='lines+markers',
            name='Лучшее значение',
            line=dict(color='green', width=3),
            marker=dict(size=6)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=generations, y=history_avg,
            mode='lines',
            name='Среднее значение',
            line=dict(color='blue', width=2, dash='dash')
        ),
        row=2, col=1
    )
    
    generations_for_plot = list(range(0, len(all_populations), max(1, len(all_populations)//10)))
    
    for i, gen in enumerate(generations_for_plot):
        pop = all_populations[gen]
        fig.add_trace(
            go.Scatter(
                x=pop[:, 0], y=pop[:, 1],
                mode='markers',
                marker=dict(size=6, 
                           color=np.linspace(0, 1, len(pop)),
                           colorscale='Viridis',
                           showscale=False),
                name=f'Поколение {gen+1}',
                visible=(i == 0)
            ),
            row=2, col=2
        )
    
    buttons = []
    for i, gen in enumerate(generations_for_plot):
        visible = [False] * (len(fig.data) - len(generations_for_plot))
        visible.extend([False] * i + [True] + [False] * (len(generations_for_plot) - i - 1))
        
        buttons.append(dict(
            label=f'Поколение {gen+1}',
            method='update',
            args=[{'visible': visible},
                  {'title': f'Популяция на поколении {gen+1}'}]
        ))
    
    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            showactive=True,
            buttons=buttons,
            x=1.05,
            xanchor='left',
            y=0.5,
            yanchor='middle'
        )],
        title=dict(
            text='Генетический алгоритм для функции Химмельблау',
            x=0.5,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='f(x,y)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        scene2=dict(
            xaxis_title='x',
            yaxis_title='y',
            xaxis=dict(range=bounds),
            yaxis=dict(range=bounds)
        ),
        height=900,
        showlegend=True,
        legend=dict(x=1.02, y=0.5)
    )
    
    fig.update_xaxes(title_text="Поколение", row=2, col=1)
    fig.update_yaxes(title_text="Значение функции", row=2, col=1)
    fig.update_xaxes(title_text="x", row=2, col=2, range=bounds)
    fig.update_yaxes(title_text="y", row=2, col=2, range=bounds)
    
    fig.show()

def plot_convergence_animation(best_point, history_best, history_avg, all_populations):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(history_best))),
        y=history_best,
        mode='lines',
        name='Лучшее значение',
        line=dict(color='green', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(len(history_avg))),
        y=history_avg,
        mode='lines',
        name='Среднее значение',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    for gen in range(len(history_best)):
        fig.add_trace(go.Scatter(
            x=[gen],
            y=[history_best[gen]],
            mode='markers',
            marker=dict(size=8, color='green'),
            name=f'Поколение {gen+1}',
            visible=False,
            text=f'Поколение: {gen+1}<br>Лучшее: {history_best[gen]:.4f}<br>Среднее: {history_avg[gen]:.4f}',
            hoverinfo='text'
        ))
    
    frames = []
    for gen in range(len(history_best)):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=list(range(gen+1)), y=history_best[:gen+1]),
                go.Scatter(x=list(range(gen+1)), y=history_avg[:gen+1]),
                go.Scatter(x=[gen], y=[history_best[gen]])
            ],
            name=f'frame_{gen}'
        ))
    
    fig.frames = frames
    
    fig.update_layout(
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 200, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 100}}],
                    "label": "▶️ Воспроизвести",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate",
                                     "transition": {"duration": 0}}],
                    "label": "⏸️ Пауза",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Поколение: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 200},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f"frame_{gen}"], {"frame": {"duration": 0, "redraw": True},
                                              "mode": "immediate",
                                              "transition": {"duration": 0}}],
                    "label": f"{gen+1}",
                    "method": "animate"
                } for gen in range(len(history_best))
            ]
        }],
        title=dict(
            text='Анимированная сходимость генетического алгоритма',
            x=0.5,
            font=dict(size=18)
        ),
        xaxis_title='Поколение',
        yaxis_title='Значение функции',
        hovermode='x unified',
        height=600
    )
    
    fig.show()

def evaluate_solution(found_point):
    distances = []
    for point in GLOBAL_MAXIMA:
        dist = np.sqrt((found_point[0] - point[0])**2 + (found_point[1] - point[1])**2)
        distances.append(dist)
    
    min_distance = min(distances)
    best_value = himmelblau(found_point[0], found_point[1])
    
    return min_distance, best_value

def create_summary_table(best_point, min_distance, best_value, time_taken, generations):
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Параметр', 'Значение'],
            fill_color='lightblue',
            align='left',
            font=dict(size=14, color='black')
        ),
        cells=dict(
            values=[
                ['Найденная точка (x, y)', 'Значение функции', 
                 'Расстояние до максимума', 'Количество поколений', 'Время выполнения'],
                [f'({best_point[0]:.6f}, {best_point[1]:.6f})', 
                 f'{best_value:.8f}', 
                 f'{min_distance:.6f}',
                 f'{generations}',
                 f'{time_taken:.2f} сек']
            ],
            fill_color='white',
            align='left',
            font=dict(size=12)
        )
    )])
    
    fig.update_layout(
        title=dict(
            text='Результаты работы генетического алгоритма',
            x=0.5,
            font=dict(size=16)
        ),
        height=300
    )
    
    fig.show()

def main():
    print("=" * 60)
    print("ГЕНЕТИЧЕСКИЙ АЛГОРИТМ С ИНТЕРАКТИВНОЙ ВИЗУАЛИЗАЦИЕЙ")
    print("Поиск максимумов функции Химмельблау")
    print("=" * 60)
    
    pop_size = 50
    generations = 30
    bounds = (-5, 5)
    
    start_time = time.time()
    
    (best_point, best_value, history_best, 
     history_avg, all_populations) = genetic_algorithm(
        pop_size=pop_size,
        generations=generations,
        bounds=bounds
    )
    
    time_taken = time.time() - start_time
    min_distance, found_value = evaluate_solution(best_point)
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ:")
    print(f"Найденная точка: x = {best_point[0]:.6f}, y = {best_point[1]:.6f}")
    print(f"Значение функции: {found_value:.8f}")
    print(f"Расстояние до ближайшего глобального максимума: {min_distance:.6f}")
    print(f"Время выполнения: {time_taken:.2f} сек")
    
    if min_distance < 0.1:
        print("\n✅ УСПЕХ: Найден один из глобальных максимумов!")
    elif min_distance < 0.5:
        print("\n⚠ ХОРОШО: Найден близко к глобальному максимуму")
    else:
        print("\n❌ МОЖНО УЛУЧШИТЬ: Попробуйте увеличить количество поколений")
    
    print("=" * 60)
    
    create_summary_table(best_point, min_distance, found_value, time_taken, generations)
    plot_interactive_results(best_point, history_best, history_avg, all_populations, bounds)
    plot_convergence_animation(best_point, history_best, history_avg, all_populations)
    
    print("\nГлобальные максимумы функции Химмельблау:")
    for i, point in enumerate(GLOBAL_MAXIMA, 1):
        x, y = point
        value = himmelblau(x, y)
        print(f"  Максимум {i}: ({x:8.4f}, {y:8.4f}) → f = {value:8.4f}")

if __name__ == "__main__":
    main()