from ple.games.flappybird import FlappyBird
from ple import PLE
from agent import Agent
from skimage.transform import resize
from skimage.morphology import disk
from skimage import img_as_ubyte
from skimage.filters.rank import gradient
from torch import load
from torch import save
import os


def preprocessing(observation):
    observation[(observation == 107)] = 0
    observation[(observation == 142)] = 0
    observation[(observation == 103)] = 0
    observation[(observation == 185)] = 0
    observation[(observation == 117)] = 0
    observation[(observation == 218)] = 0
    observation[(observation == 128)] = 0
    noisy_image = img_as_ubyte(observation)
    # Edge detection
    grad = gradient(noisy_image, disk(5))
    return resize(grad, (80, 80))


os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True, force_fps=False)

agent = Agent(allowed_actions=p.getActionSet(), channels=1, learning_rate=0.0085)
try:
    agent.model.load_state_dict(load('memento_movement.pt'))
except EOFError:
    print("Error loading the saved model state")


p.init()

nb_frames = 10000000
rewards = []
episode = []
old_observation = preprocessing(p.getScreenGrayscale())
movement_captioning = 0
for i in range(nb_frames):
    if p.game_over():
        p.reset_game()
    preprocessed_observation = preprocessing(p.getScreenGrayscale())
    # Forward
    action, log_action = agent.pickAction(preprocessed_observation-old_observation)
    if movement_captioning < 2:
        old_observation = preprocessed_observation
        movement_captioning += 1
    elif movement_captioning < 15:
        movement_captioning += 1
    else:
        movement_captioning = 0
    reward_action = p.act(action)
    rewards += [reward_action]
    episode += [log_action]
    if reward_action != 0:
        agent.train(rewards, episode)
        save(agent.model.state_dict(), 'memento_movement.pt')
        rewards = []
        episode = []
