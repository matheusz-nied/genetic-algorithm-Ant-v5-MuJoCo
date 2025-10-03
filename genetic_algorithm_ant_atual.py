from random import random
import gymnasium as gym
import numpy as np
import os
import yaml
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt

# Set the MUJOCO_GL environment variable to 'egl' for hardware acceleration
os.environ["MUJOCO_GL"] = "egl" # REQUIRED FOR RECORDING

def evaluate_ant_with_sine_waves(params, render=False, record_video=False, video_folder="videos", video_name="ant_video"):
    """
    Evaluates the Ant environment using sine wave controllers.
    
    Args:
        params (list): List of 17 parameters:
            - First parameter: Frequency (shared by all joints)
            - Next 8 parameters: Amplitudes of sine waves (0 to 1)
            - Last 8 parameters: Phase shifts of sine waves (0 to 2π)
        render (bool): Whether to render the environment
        record_video (bool): Whether to record a video
        video_folder (str): Folder to save the video
        video_name (str): Name of the video file
        
    Returns:
        float: Total reward accumulated
    """
    # Validate parameters
    if len(params) != 17:
        raise ValueError("Esperado 17 parâmetros (1 frequência, 8 amplitudes, e 8 fases)")
    
    # Extract frequency (single value), amplitudes, and phase shifts
    frequency = params[0]
    amplitudes = params[1:9]
    phase_shifts = params[9:]
    
    # Ensure amplitudes are between 0 and 1
    amplitudes = np.clip(amplitudes, 0, 1)
    try:
        env = gym.make("Ant-v5", render_mode="rgb_array" if render or record_video else None, terminate_when_unhealthy=False)
        
        if record_video:
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            env = RecordVideo(env, video_folder, name_prefix=video_name, episode_trigger=lambda x: True)
        
        observation, _ = env.reset()
        
        total_reward = 0
        timesteps = 1000
        
        # Run simulation
        for t in range(timesteps):
            # Generate actions using sine waves with phase shifts
            time_factor = t / 20.0  # Scale time for reasonable frequencies
            actions = np.array([
                amplitudes[i] * np.sin(frequency * time_factor + phase_shifts[i])
                for i in range(8)
            ])
            
            # Take action in environment
            observation, reward, terminated, truncated, info = env.step(actions)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        env.close()
        return total_reward
    except Exception as e:
        print(f"Ocorreu um erro durante a avaliação: {e}")
        return -1000 # Retorna uma fitness baixa em caso de erro

def plot_sine_waves(params, timesteps=1000, filename="best_solution_waves.png", separate_plots=False):
    """
    Plot the sine waves used for controlling the ant
    
    Args:
        params (list): List of 17 parameters (1 frequency, 8 amplitudes, and 8 phase shifts)
        timesteps (int): Number of timesteps to plot
        separate_plots (bool): If True, plot each sine wave in a separate subplot
    """

    if params is None or len(params) == 0:
        print("Nenhum parâmetro fornecido para plotagem.")
        return
        
    frequency = params[0]
    amplitudes = params[1:9]
    phase_shifts = params[9:]
    
    # Ensure amplitudes are between 0 and 1
    amplitudes = np.clip(amplitudes, 0, 1)

    # Create time array with the same time_factor used in evaluate_ant_with_sine_waves
    time = np.arange(timesteps) / 20.0  # Scale time for reasonable frequencies
    
    if separate_plots:
        # Create a figure with 8 subplots (2x4 grid)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
        axes = np.atleast_1d(axes).flatten()

        # Plot each sine wave in its own subplot
        for i in range(8):
            actions = amplitudes[i] * np.sin(frequency * time + phase_shifts[i])
            axes[i].plot(time, actions)
            axes[i].set_title(f"Joint {i+1}: f={frequency:.2f}, A={amplitudes[i]:.2f}, φ={phase_shifts[i]:.2f}")
            axes[i].grid(True)

        # Set common labels with proper padding
        fig.text(0.5, 0.01, "Time (scaled by factor 1/20)", ha='center')
        fig.text(0.04, 0.5, "Joint Action", va='center', rotation='vertical')
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # Adjust the layout to leave room for labels
        plt.savefig(f"{filename}_individual_plots.png")
    else:
        # Create a single figure with all sine waves
        plt.figure(figsize=(12, 8))
        
        # Plot each sine wave
        for i in range(8):
            actions = amplitudes[i] * np.sin(frequency * time + phase_shifts[i])
            plt.plot(time, actions, label=f"Joint {i+1}: f={frequency:.2f}, A={amplitudes[i]:.2f}, φ={phase_shifts[i]:.2f}")
        
        plt.title("Sine Wave Control Signals for Ant Joints")
        plt.xlabel("Time (scaled by factor 1/20)")
        plt.ylabel("Joint Action")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{filename}_combined_plot.png")

def plot_fitness_evolution(best_fitness_history, avg_fitness_history, filename="fitness_evolution.png"):
    """
    Plota a evolução do fitness ao longo das gerações
    
    Args:
        best_fitness_history (list): Lista com os melhores fitness de cada geração
        avg_fitness_history (list): Lista com os fitness médios de cada geração
        filename (str): Nome do arquivo para salvar o gráfico
    """
    plt.figure(figsize=(12, 8))
    generations = range(1, len(best_fitness_history) + 1)
    
    plt.plot(generations, best_fitness_history, 'b-', label='Melhor Fitness', linewidth=2)
    plt.plot(generations, avg_fitness_history, 'r-', label='Fitness Médio', linewidth=2)
    
    plt.title('Evolução do Fitness ao Longo das Gerações')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adiciona valores máximo e mínimo no gráfico
    max_fitness = max(best_fitness_history)
    min_fitness = min(avg_fitness_history)
    plt.annotate(f'Max: {max_fitness:.2f}', 
                xy=(generations[best_fitness_history.index(max_fitness)], max_fitness),
                xytext=(10, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='blue'))
    
    plt.savefig(filename)
    plt.close()

def plot_fitness_analysis(all_fitnesses_history, results_folder):
    """
    Gera gráficos de análise da distribuição do fitness:
    - Desvio padrão por geração
    - Histogramas do fitness em algumas gerações
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    num_generations = len(all_fitnesses_history)

    # --- Gráfico do desvio padrão ---
    stds = [np.std(f) for f in all_fitnesses_history]
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_generations+1), stds, 'g-', linewidth=2)
    plt.title("Desvio Padrão do Fitness por Geração")
    plt.xlabel("Geração")
    plt.ylabel("Desvio Padrão do Fitness")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_folder, "fitness_std.png"))
    plt.close()

    # --- Histogramas de algumas gerações ---
    sample_gens = [0, num_generations//4, 2*num_generations//4, 3*num_generations//4, num_generations-1]
    for gen in sample_gens:
        plt.figure(figsize=(8, 6))
        plt.hist(all_fitnesses_history[gen], bins=20, color="blue", alpha=0.7)
        plt.title(f"Distribuição de Fitness - Geração {gen+1}")
        plt.xlabel("Fitness")
        plt.ylabel("Frequência")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_folder, f"fitness_hist_gen_{gen+1}.png"))
        plt.close()

def save_parameters(params, config, best_fitness, filename="parameters.txt"):
    """
    Salva os parâmetros e configurações em um arquivo de texto
    
    Args:
        params (list): Lista de parâmetros da melhor solução
        config (dict): Configurações do algoritmo genético
        best_fitness (float): Melhor fitness alcançado
        filename (str): Nome do arquivo para salvar
    """
    with open(filename, 'w') as f:
        f.write("=== PARÂMETROS DA MELHOR SOLUÇÃO ===\n")
        f.write(f"Melhor Fitness: {best_fitness}\n\n")
        
        # Representação: 1 frequência + 8 amplitudes + 8 fases = 17 parâmetros
        f.write("Frequência (compartilhada por todas as juntas):\n")
        f.write(f"  {params[0]:.4f} Hz\n")
        
        f.write("\nAmplitudes (8 juntas):\n")
        for i, amp in enumerate(params[1:9]):
            f.write(f"  Junta {i+1}: {amp:.4f}\n")
        
        f.write("\nFases (8 juntas):\n")
        for i, phase in enumerate(params[9:17]):
            f.write(f"  Junta {i+1}: {phase:.4f}\n")
        
        f.write("\n=== CONFIGURAÇÕES DO ALGORITMO GENÉTICO ===\n")
        for key, value in config.items():
            if isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue}\n")
            else:
                f.write(f"{key}: {value}\n")

def create_results_folder():
    """
    Cria uma pasta enumerada para armazenar os resultados
    
    Returns:
        str: Caminho da pasta criada
    """
    base_folder = "graficos"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    # Encontra o próximo número disponível
    i = 0
    while os.path.exists(os.path.join(base_folder, str(i))):
        i += 1
    
    results_folder = os.path.join(base_folder, str(i))
    os.makedirs(results_folder)
    
    return results_folder

# --- IMPLEMENTAÇÃO DO ALGORITMO GENÉTICO (Funções de suporte) ---
def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def create_individual(config):
    freq_bounds = config['bounds']['frequency']
    amp_bounds = config['bounds']['amplitude']
    phase_bounds = config['bounds']['phase']
    # Agora apenas 1 frequência compartilhada
    frequency = np.random.uniform(freq_bounds[0], freq_bounds[1])
    amplitudes = np.random.uniform(amp_bounds[0], amp_bounds[1], size=8)
    phases = np.random.uniform(phase_bounds[0], phase_bounds[1], size=8)
    return np.concatenate([[frequency], amplitudes, phases])

def create_initial_population(config):
    return [create_individual(config) for _ in range(config['population_size'])]

def calculate_fitness(individual):
    return evaluate_ant_with_sine_waves(individual)

def selection(population, fitnesses, config):
    tournament_size = config['tournament_size']
    indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_individuals = [population[i] for i in indices]
    tournament_fitnesses = [fitnesses[i] for i in indices]
    winner_index = np.argmax(tournament_fitnesses)
    return tournament_individuals[winner_index]

def blx_alpha_crossover(parent1, parent2, alpha=0.5):
    """
    BLX-α crossover que permite extrapolação além dos limites dos pais.
    
    Args:
        parent1, parent2: Os pais para cruzamento
        alpha: Parâmetro que controla a extensão da extrapolação (0.3-0.5 são valores típicos)
    """
    child1, child2 = parent1.copy(), parent2.copy()
    
    for i in range(len(parent1)):
        # Encontra o mínimo e máximo entre os pais
        min_val = min(parent1[i], parent2[i])
        max_val = max(parent1[i], parent2[i])
        
        # Calcula o intervalo estendido
        range_val = max_val - min_val
        extended_min = np.max(min_val - alpha * range_val, 0)
        extended_max = max_val + alpha * range_val
        
        # Gera filhos no intervalo estendido
        child1[i] = np.random.uniform(extended_min, extended_max)
        child2[i] = np.random.uniform(extended_min, extended_max)
    
    return child1, child2

def arithmetic_crossover(parent1, parent2, config):
    """
    Realiza o crossover aritmético ponderado, usando um único alpha
    para cada bloco de parâmetros (frequências, amplitudes, fases).
    """
    child1, child2 = parent1.copy(), parent2.copy()
    if np.random.rand() < config['crossover_rate']:
        # Gera três valores alpha distintos, um para cada tipo de parâmetro.
        alpha_freqs = np.random.rand() * 1.5
        alpha_amps = np.random.rand()
        alpha_phases = np.random.rand()

        # Aplica o crossover para os blocos
        child1[:1] = alpha_freqs * parent1[:1] + (1 - alpha_freqs) * parent2[:1]
        child2[:1] = alpha_freqs * parent2[:1] + (1 - alpha_freqs) * parent1[:1]

        child1[1:9] = alpha_amps * parent1[1:9] + (1 - alpha_amps) * parent2[1:9]
        child2[1:9] = alpha_amps * parent2[1:9] + (1 - alpha_amps) * parent1[1:9]

        child1[9:] = alpha_phases * parent1[9:] + (1 - alpha_phases) * parent2[9:]
        child2[9:] = alpha_phases * parent2[9:] + (1 - alpha_phases) * parent1[9:]

    return child1, child2

def mutation(individual, current_mutation_rate, config):
    """
    Aplica mutação gaussiana.
    - Gene 0: Frequência única (compartilhada por todas as juntas)
    - Genes 1-8: Amplitudes
    - Genes 9-16: Fases
    """
    mutation_strength = config['mutation_strength']
    bounds = config['bounds']
    mutated_individual = individual.copy()

    # --- Mutação da Frequência Única (gene 0) ---
    if np.random.rand() < current_mutation_rate:
        bound = bounds['frequency']
        range_width = bound[1] - bound[0]
        mutation_value = np.random.normal(0, mutation_strength * range_width)
        mutated_individual[0] += mutation_value
        mutated_individual[0] = np.clip(mutated_individual[0], bound[0], bound[1])

    # --- Mutação Individual para Amplitude e Fase ---
    for i in range(1, len(mutated_individual)):
        if np.random.rand() < current_mutation_rate:
            if i < 9: bound = bounds['amplitude']  # Genes 1-8
            else: bound = bounds['phase']           # Genes 9-16
            
            range_width = bound[1] - bound[0]
            mutation_value = np.random.normal(0, mutation_strength * range_width)
            mutated_individual[i] += mutation_value
            mutated_individual[i] = np.clip(mutated_individual[i], bound[0], bound[1])

    return mutated_individual

# --- FUNÇÃO PRINCIPAL DO AG ---
def run_genetic_algorithm():
    config = load_config()
    
    # Cria pasta para resultados
    results_folder = create_results_folder()
    print(f"Resultados serão salvos em: {results_folder}")

    population = create_initial_population(config)
    best_overall_individual = None
    best_overall_fitness = -np.inf
    
    # Históricos para plotagem
    best_fitness_history = []
    avg_fitness_history = []
    all_fitnesses_history = []

    # NOVO: Variáveis para controle dos novos critérios
    convergence_counter = 0
    stagnation_counter = 0
    current_mutation_rate = config['mutation_rate'] # Inicializa a taxa de mutação
    
    print("Iniciando o Algoritmo Genético...")
    print(f"Critério de parada por convergência: {'Ativado' if config['enable_convergence_stop'] else 'Desativado'}")
    print(f"Mutação adaptativa (taxa): {'Ativada' if config['enable_adaptive_mutation'] else 'Desativada'}")
    print(f"Mutação adaptativa (força): {'Ativada' if config.get('enable_adaptive_strength', False) else 'Desativada'}")

    for generation in range(config['num_generations']):
        print(f"\n Tamanho da população: {len(population)}")
        print(f"\n--- Geração {generation + 1}/{config['num_generations']} ---")
        
        fitnesses = [calculate_fitness(ind) for ind in tqdm(population, desc="Avaliando Fitness")]
        
        # Métricas da geração
        avg_fitness = np.mean(fitnesses)
        best_gen_fitness = np.max(fitnesses)
        best_gen_idx = np.argmax(fitnesses)
        
        # Salva no histórico
        best_fitness_history.append(best_gen_fitness)
        avg_fitness_history.append(avg_fitness)
        all_fitnesses_history.append(fitnesses)

        # Salva o melhor fitness geral anterior para checar estagnação
        last_best_fitness = best_overall_fitness 
        
        if best_gen_fitness > best_overall_fitness:
            best_overall_fitness = best_gen_fitness
            best_overall_individual = population[best_gen_idx]
            print(f"Novo melhor fitness: {best_overall_fitness:.4f} | Fitness médio: {avg_fitness:.4f}")
        else:
            print(f"Melhor da geração: {best_gen_fitness:.4f} | Melhor geral: {best_overall_fitness:.4f} | Fitness médio: {avg_fitness:.4f}")

        # NOVO: Mutação Adaptativa de Força (Linear)
        if config.get('enable_adaptive_strength', False):
            # Decaimento linear de mutation_strength
            progress = generation / (config['num_generations'] - 1)  # 0.0 a 1.0
            initial_strength = config['initial_mutation_strength']
            final_strength = config['final_mutation_strength']
            current_mutation_strength = initial_strength - (initial_strength - final_strength) * progress
            # Atualiza o config temporariamente para esta geração
            config['mutation_strength'] = current_mutation_strength
            print(f"Força de mutação atual: {current_mutation_strength:.4f}")
        
        # NOVO: Lógica de Mutação Adaptativa (Taxa)
        if config['enable_adaptive_mutation']:
            # Verifica estagnação
            if best_overall_fitness <= last_best_fitness:
                stagnation_counter += 1
            else:
                stagnation_counter = 0 # Reset se houver melhora

            # Se estagnado, aumenta a mutação
            if stagnation_counter >= config['stagnation_generations']:
                current_mutation_rate *= config['mutation_increase_factor']
                current_mutation_rate = min(current_mutation_rate, config['max_mutation_rate'])
                print(f"População estagnada! Aumentando taxa de mutação para: {current_mutation_rate:.4f}")
                stagnation_counter = 0 # Reset para dar tempo da nova taxa agir
            else:
                # Se não estagnado, aplica decaimento lento
                current_mutation_rate *= config['mutation_decrease_factor']
                current_mutation_rate = max(current_mutation_rate, config['min_mutation_rate'])
            
            print(f"Taxa de mutação atual: {current_mutation_rate:.4f}")

        # NOVO: Lógica do Critério de Parada por Convergência
        if config['enable_convergence_stop']:
            # Evita divisão por zero e funciona com fitness negativo
            if abs(best_overall_fitness) > 1e-6:
                diff = (best_overall_fitness - avg_fitness) / abs(best_overall_fitness)
                if diff < config['convergence_threshold']:
                    convergence_counter += 1
                else:
                    convergence_counter = 0 # Reset se a população divergir novamente
            
            print(f"Gerações de convergência consecutivas: {convergence_counter}/{config['convergence_generations']}")

            if convergence_counter >= config['convergence_generations']:
                print("\nPopulação convergiu! Encerrando a evolução.")
                break

        # Criação da Próxima Geração
        next_generation = []
        sorted_indices = np.argsort(fitnesses)[::-1]
        for i in range(config['elitism_count']):
            next_generation.append(population[sorted_indices[i]])

        while len(next_generation) < config['population_size']:
            parent1 = selection(population, fitnesses, config)
            parent2 = selection(population, fitnesses, config)

            if np.random.random() < config['arithmetic_crossover_rate']:
                child1, child2 = arithmetic_crossover(parent1, parent2, config)
            else:
                child1, child2 = blx_alpha_crossover(parent1, parent2)
            
            # Passa a taxa de mutação atual para a função
            child1 = mutation(child1, current_mutation_rate, config)
            child2 = mutation(child2, current_mutation_rate, config)
            
            next_generation.append(child1)
            if len(next_generation) < config['population_size']:
                next_generation.append(child2)
        
        population = next_generation
        
    # --- Fim do Algoritmo ---
    print("\n--- Evolução Concluída ---")
    print(f"Melhor fitness final: {best_overall_fitness}")
    print("Melhores parâmetros encontrados:")
    print(best_overall_individual)
    
    # Salva todos os resultados na pasta criada
    print(f"\nSalvando resultados na pasta: {results_folder}")
    
    # Gera e salva os gráficos
    plot_fitness_evolution(best_fitness_history, avg_fitness_history, 
                          os.path.join(results_folder, "fitness_evolution.png"))
    plot_sine_waves(best_overall_individual, 
                   filename=os.path.join(results_folder, "sine_waves_combined.png"))
    plot_sine_waves(best_overall_individual, separate_plots=True,
                   filename=os.path.join(results_folder, "sine_waves_individual.png"))
    plot_fitness_analysis(all_fitnesses_history, results_folder)
    
    # Salva os parâmetros
    save_parameters(best_overall_individual, config, best_overall_fitness,
                   os.path.join(results_folder, "parameters.txt"))
    
    # Gera o vídeo
    video_folder = os.path.join(results_folder, "videos")
    print("\nGerando vídeo da melhor solução...")
    evaluate_ant_with_sine_waves(best_overall_individual, render=True, record_video=True, 
                                video_folder=video_folder, video_name="best_ant_solution")
    print("Vídeo salvo na pasta 'videos'.")
    
    print(f"\nTodos os resultados foram salvos em: {results_folder}")


if __name__ == "__main__":
    run_genetic_algorithm()