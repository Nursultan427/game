import pygame
import numpy as np
import random
import matplotlib.pyplot as plt

# -------------------
# настройки
# -------------------
WIDTH, HEIGHT = 600, 600
CELL = 40
GRID = WIDTH // CELL

# цвета
WHITE = (255,255,255)
BLUE = (50,100,255)
GREEN = (0,255,0)
RED = (255,50,50)
YELLOW = (255,255,0)

ACTIONS = [0,1,2,3] 

# Q-table: (dx, dy, danger_up, danger_down, danger_left, danger_right, action)
q_table = {}

alpha = 0.1
gamma = 0.9
epsilon = 0.2

rewards = []

# -------------------
# функции
# -------------------
def get_state(player, coin, enemies):
    px, py = player
    cx, cy = coin

    dx = np.sign(cx - px)
    dy = np.sign(cy - py)

    danger = [0,0,0,0]

    for ex, ey in enemies:
        if ex == px and ey == py-1: danger[0] = 1
        if ex == px and ey == py+1: danger[1] = 1
        if ex == px-1 and ey == py: danger[2] = 1
        if ex == px+1 and ey == py: danger[3] = 1

    return (dx, dy, *danger)


def choose_action(state):
    if state not in q_table:
        q_table[state] = np.zeros(4)

    if random.random() < epsilon:
        return random.choice(ACTIONS)

    return np.argmax(q_table[state])


def update_q(state, action, reward, next_state):
    if next_state not in q_table:
        q_table[next_state] = np.zeros(4)

    best_next = np.max(q_table[next_state])

    q_table[state][action] += alpha * (
        reward + gamma * best_next - q_table[state][action]
    )


def move(pos, action):
    x, y = pos

    if action == 0: y -= 1
    elif action == 1: y += 1
    elif action == 2: x -= 1
    elif action == 3: x += 1

    x = max(0, min(GRID-1, x))
    y = max(0, min(GRID-1, y))

    return (x, y)


# -------------------
# обучение
# -------------------
NUM_EPISODES = 1000
MAX_STEPS = 200
for episode in range(NUM_EPISODES):

    player = (random.randint(0, GRID-1), random.randint(0, GRID-1))
    coin = (random.randint(0, GRID-1), random.randint(0, GRID-1))
    enemies = [(random.randint(0, GRID-1), random.randint(0, GRID-1))]

    total_reward = 0

    for step in range(MAX_STEPS):

        state = get_state(player, coin, enemies)
        action = choose_action(state)

        new_player = move(player, action)

        reward = -1

        
        if new_player == coin:
            reward = 20
            coin = (random.randint(0, GRID-1), random.randint(0, GRID-1))

        # враг
        if new_player in enemies:
            reward = -50
            break

        next_state = get_state(new_player, coin, enemies)

        update_q(state, action, reward, next_state)

        player = new_player
        total_reward += reward

    rewards.append(total_reward)

# график обучения
window = 50
smoothed_rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
learn_threshold = 5
learned_episode = None
for idx, value in enumerate(smoothed_rewards, start=window):
    if value >= learn_threshold:
        learned_episode = idx
        break

plt.figure(figsize=(14, 8))
plt.plot(rewards, alpha=0.25, color='gray', label='Награда за эпизод')
plt.plot(range(window - 1, len(rewards)), smoothed_rewards, color='blue', linewidth=3, label=f'Скользящее среднее ({window})')
if learned_episode is not None:
    plt.axvline(learned_episode, color='red', linestyle='--', linewidth=1.5, label=f'Порог изучения ~{learned_episode} эп.')
    plt.text(learned_episode + 5, max(smoothed_rewards) * 0.9, f'Эпизод {learned_episode}', color='red')

plt.xlabel('Эпизод')
plt.ylabel('Общая награда')
plt.title('График обучения агента по эпизодам')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
print('Открываю график обучения. Закройте окно графика, чтобы запустить игру.')
plt.show(block=True)

if learned_episode is not None:
    print(f"Агент стабильно начал получать среднюю награду >= {learn_threshold} примерно на эпизоде {learned_episode}.")
else:
    print("Агент пока не достиг порога средней награды за эпизод.")

print('График закрыт. Запускаю игровое окно RL Arena...')
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Arena")
clock = pygame.time.Clock()

player = (5,5)
coin = (10,10)
enemies = [(3,3)]

running = True

while running:
    clock.tick(8)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = get_state(player, coin, enemies)
    action = choose_action(state)

    player = move(player, action)

    # движение врага (рандом)
    enemies = [move(e, random.choice(ACTIONS)) for e in enemies]

    if player == coin:
        coin = (random.randint(0, GRID-1), random.randint(0, GRID-1))

    if player in enemies:
        print("Game Over")
        running = False

    # отрисовка
    screen.fill(WHITE)

    for x in range(GRID):
        for y in range(GRID):
            pygame.draw.rect(screen, (220,220,220), (x*CELL,y*CELL,CELL,CELL),1)

    pygame.draw.rect(screen, BLUE, (player[0]*CELL, player[1]*CELL, CELL, CELL))
    pygame.draw.rect(screen, GREEN, (coin[0]*CELL, coin[1]*CELL, CELL, CELL))

    for e in enemies:
        pygame.draw.rect(screen, RED, (e[0]*CELL, e[1]*CELL, CELL, CELL))

    pygame.display.flip()

pygame.quit()
plt.show()