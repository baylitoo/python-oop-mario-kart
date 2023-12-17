MAX_ANGLE_VELOCITY = 0.05
BLOCK_SIZE = 50
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pygame
import math
import random

N = 19  # Distance entre kart et checkpoint, orientation à suivre, vitesse, surface 4X4 array (one hot encoding)


# Policy gradiant NN
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the PolicyNetwork with specified input size, hidden layer size, and output size
        super(PolicyNetwork, self).__init__()

        # Define layers of the neural network
        self.__fc1 = nn.Linear(input_size, hidden_size)   # First hidden layer
        self.__fc2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
        # Additional hidden layers
        self.__fc3 = nn.Linear(hidden_size, hidden_size)
        self.__fc4 = nn.Linear(hidden_size, hidden_size)
        self.__fc5 = nn.Linear(hidden_size, hidden_size)
        self.__fc6 = nn.Linear(hidden_size, hidden_size)
        self.__fc7 = nn.Linear(hidden_size, output_size)  # Output layer with softmax activation

    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.__fc1(x))  # ReLU activation for first layer
        # ReLU activation for subsequent layers
        x = F.relu(self.__fc2(x))
        x = F.relu(self.__fc3(x))
        x = F.relu(self.__fc4(x))
        x = F.relu(self.__fc5(x))
        x = F.relu(self.__fc6(x))
        x = F.softmax(self.__fc7(x), dim=0)  # Softmax activation for output
        return x


# AI Class using the Policy NN
class AITraining:
    def __init__(self, policy_network, string):
        # Initialize the AI training class
        self.__checkpoints_visited = {'C': False, 'D': False, 'E': False, 'F': False}  # Track checkpoints visited
        self.kart = None  # Reference to the kart object
        self.__Name = "AI"  # Name of the AI controller
        self.__policy_network = policy_network  # Policy network used for decision making
        self.__optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.0001)  # Optimizer for the policy network
        self.__checkpoints_positions = {}  # Stores the positions of checkpoints
        self.__states = []  # List of observed states
        self.__actions = []  # List of actions taken
        self.__current_surfaces = ["R"]  # List of current surface types
        self.__next_checkpoint_ids = [0]  # List of next checkpoint IDs
        self.__track_string = string  # String representation of the track
        self.__surrounding_surfacess = []  # List of surrounding surfaces for each state
        self.__full_states = []  # Full state information including surroundings
        self.__epsilon = 0.1  # Initial exploration rate
        self.__decay_rate = 0.99  # Decay rate for exploration

    @property
    def name(self):
        # Getter for the AI name
        return self.__Name

    def convert_surfaces(self, surfaces):
        # Convert surface types to one-hot encoded vectors
        surface_mapping = {
            'R': [1, 0, 0, 0],  # Road
            'G': [0, 1, 0, 0],  # Grass
            'L': [0, 0, 1, 0],  # Lava
            'B': [0, 0, 0, 1],  # Boost
            'Unknown': [0, 0, 0, 0]  # Unknown or other types
        }
        return [surface_mapping.get(surface, surface_mapping['Unknown']) for surface in surfaces]

    def update_epsilon(self, decay_rate):
        # Reduce the exploration rate over time
        self.__epsilon = max(self.__epsilon * decay_rate, 0.01)

    def move(self, string):
        # Determine the next move of the kart based on the current state
        state, surrounding_surfaces = self.kart.get_state(self.__track_string)
        numerical_surfaces = self.convert_surfaces(surrounding_surfaces)
        flattened_surfaces = [num for sublist in numerical_surfaces for num in sublist]
        flattened_surfaces_array = np.array(flattened_surfaces, dtype=np.float32)
        full_state = np.concatenate((state, flattened_surfaces_array))
        if random.random() < self.__epsilon:  # Exploration phase
            self.__action_index = random.randint(0, 3)
        else:  # Exploitation phase
            action_probs = self.__policy_network(torch.from_numpy(full_state).float())
            self.__action_index = torch.multinomial(action_probs, 1).item()

        self.__surrounding_surfacess.append(surrounding_surfaces)
        self.__states.append(state)
        self.__full_states.append(full_state)
        self.__actions.append(self.__action_index)
        if self.kart.current_surface is not None:
            self.__current_surfaces.append(self.kart.current_surface)
            self.__next_checkpoint_ids.append(self.kart._next_checkpoint_id)

        command = [False, False, False, False]
        key_list = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]
        command[self.__action_index] = True
        keys = {key: command[i] for i, key in enumerate(key_list)}
        return keys

    def __calculate_returns(self, rewards, gamma):
        # Calculate the returns for each time step
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        return returns

    def __update_policy(self, states, actions, returns):
        # Update the policy network based on observed states, actions, and returns
        policy_loss = []
        for state, action, G in zip(states, actions, returns):
            state = torch.from_numpy(np.array(state, dtype=np.float32))
            action_probs = self.__policy_network(state)
            log_prob = torch.log(action_probs[action])
            policy_loss.append(-log_prob * G)

        policy_loss = [loss.unsqueeze(0) for loss in policy_loss]

        self.__optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()

        policy_loss.backward()
        self.__optimizer.step()

    def train(self):
        # Training process for the AI agent
        rewards = [self.__calculate_reward(self.__states[i], self.__states[i + 1], self.__current_surfaces[i],
                                           self.__surrounding_surfacess[i], self.__surrounding_surfacess[i + 1],
                                           self.__next_checkpoint_ids[i])
                   for i in range(len(self.__states) - 1)]

        self.__states.pop()
        self.__actions.pop()
        self.__full_states.pop()
        self.__current_surfaces.pop()
        self.__next_checkpoint_ids.pop()

        returns = self.__calculate_returns(rewards, gamma=0.99)
        self.__update_policy(self.__full_states, self.__actions, returns)
        self.update_epsilon(self.__decay_rate)
        self.__full_states = []
        self.__states = []
        self.__actions = []
        self.__current_surfaces = []

    def __calculate_reward(self, state, new_state, current_surface, old_surrounding_surfaces, new_surrounding_surfaces,
                           next_checkpoint_id):
        # Calculate the reward for the AI agent based on the current and new state

        reward = 0
        checkpoint_id_map = {'C': 0, 'D': 1, 'E': 2, 'F': 3}

        old_distance_to_checkpoint, old_angle_to_checkpoint, old_velocity = state
        new_distance_to_checkpoint, new_angle_to_checkpoint, new_velocity = new_state

        # Reward for reducing the distance to the checkpoint
        distance_reduction = old_distance_to_checkpoint - new_distance_to_checkpoint
        if distance_reduction == 0:
            reward = -40
        else:
            reward += 12000 * distance_reduction

        # Compare old and new surrounding surfaces and adjust the reward accordingly
        for old_surface, new_surface in zip(old_surrounding_surfaces, new_surrounding_surfaces):
            if old_surface in ['L', 'G'] and new_surface in ['R', 'B']:
                reward += 300  # Reward for moving to the road or boost
            elif old_surface in ['R', 'B'] and new_surface in ['L', 'G']:
                reward -= 500  # Penalty for moving to grass or lava

        # Velocity-based reward
        if new_velocity == 0:
            reward += -10
        else:
            velocity_reward = new_velocity - old_velocity
            reward += 10 * velocity_reward

        # Angle-based reward
        if -math.pi / 10 <= new_angle_to_checkpoint <= math.pi / 10:
            reward += 60
        else:
            reward += -60

        # Surface-based rewards and penalties
        surface_rewards = {
            "R": 2,  # Reward for being on the road
            "B": 3,  # Reward for being on a boost
            "G": -5,  # Penalty for being on grass
            "L": -20,  # Penalty for being on lava
        }

        checkpoint_penalty = -10
        checkpoint_rewards = {
            "C": 10,  # Reward for checkpoint C
            "D": 20,  # Reward for checkpoint D
            "E": 30,  # Reward for checkpoint E
            "F": 40  # Reward for checkpoint F
        }

        if current_surface in checkpoint_id_map:
            if checkpoint_id_map[current_surface] == next_checkpoint_id:
                if self.__checkpoints_visited[current_surface]:
                    # Apply penalty for revisiting checkpoint
                    reward += checkpoint_penalty
                else:
                    # First visit to the checkpoint
                    reward += checkpoint_rewards[current_surface]
                    self.__checkpoints_visited[current_surface] = True

            else:
                reward += checkpoint_penalty


        else:
            # Default reward/penalty for other surfaces
            reward += surface_rewards.get(current_surface, 0)

        # Get track dimensions
        track_lines = self.__track_string.split('\n')
        track_height = len(track_lines)
        track_width = len(track_lines[0]) if track_height > 0 else 0

        # Penalize if kart is out of the track
        out_of_bounds_penalty = -40  # double of lava
        if self.kart._x < 0 or self.kart._y < 0 or self.kart._x > track_width * BLOCK_SIZE or self.kart._y > track_height * BLOCK_SIZE:
            reward += out_of_bounds_penalty

        return reward

    def save_model(self, filepath):
        """
        Sauvegarde le modèle entraîné dans un fichier.

        :param filepath: Chemin du fichier où le modèle sera sauvegardé.
        """
        torch.save(self.__policy_network.state_dict(), filepath)
        print(f"Modèle sauvegardé dans {filepath}")

    def load_model(self, filepath):
        """
        Charge un modèle à partir d'un fichier.

        :param filepath: Chemin du fichier à partir duquel charger le modèle.
        """
        self.__policy_network.load_state_dict(torch.load(filepath))
        self.__policy_network.eval()  # Mettre le modèle en mode évaluation
        print(f"Modèle chargé depuis {filepath}")


class AI():

    def __init__(self):
        self.kart = None

    def set_kart(self, kart):
        self.kart = kart

    def move(self, string):
        """
        Cette methode contient une implementation d'IA tres basique.
        L'IA identifie la position du prochain checkpoint et se dirige dans sa direction.

        :param string: La chaine de caractere decrivant le circuit
        :param screen: L'affichage du jeu
        :param position: La position [x, y] actuelle du kart
        :param angle: L'angle actuel du kart
        :param velocity: La vitesse [vx, vy] actuelle du kart
        :param next_checkpoint_id: Un entier indiquant le prochain checkpoint a atteindre
        :returns: un tableau de 4 boolean decrivant quelles touches [UP, DOWN, LEFT, RIGHT] activer
        """

        # =================================================
        # D'abord trouver la position du checkpoint
        # =================================================
        if self.kart.next_checkpoint_id == 0:
            char = 'C'
        elif self.kart.next_checkpoint_id == 1:
            char = 'D'
        elif self.kart.next_checkpoint_id == 2:
            char = 'E'
        elif self.kart.next_checkpoint_id == 3:
            char = 'F'

        # On utilise x et y pour decrire les coordonnees dans la chaine de caractere
        # x indique le numero de colonne
        # y indique le numero de ligne
        x, y = 0, 0
        for c in string:

            # Si on trouve le caractere correpsondant au checkpoint, on s'arrete
            if c == char:
                break

            # Si on trouve le caractere de retour a la ligne "\n" on incremente y et on remet x a 0
            # Sinon on incremente x
            if c == "\n":
                y += 1
                x = 0
            else:
                x += 1

        next_checkpoint_position = [x * BLOCK_SIZE + .5 * BLOCK_SIZE, y * BLOCK_SIZE + .5 * BLOCK_SIZE]

        # =================================================
        # Ensuite, trouver l'angle vers le checkpoint
        # =================================================
        relative_x = next_checkpoint_position[0] - self.kart.position[0]
        relative_y = next_checkpoint_position[1] - self.kart.position[1]

        # On utilise la fonction arctangente pour calculer l'angle du vecteur [relative_x, relative_y]
        next_checkpoint_angle = math.atan2(relative_y, relative_x)

        # L'angle relatif correspond a la rotation que doit faire le kart pour se trouver face au checkpoint
        # On applique l'operation (a + pi) % (2*pi) - pi pour obtenir un angle entre -pi et pi
        relative_angle = (next_checkpoint_angle - self.kart.angle + math.pi) % (2 * math.pi) - math.pi

        # =================================================
        # Enfin, commander le kart en fonction de l'angle
        # =================================================
        if relative_angle > MAX_ANGLE_VELOCITY:
            # On tourne a droite
            command = [False, False, False, True]
        elif relative_angle < -MAX_ANGLE_VELOCITY:
            # On tourne a gauche
            command = [False, False, True, False]
        else:
            # On avance
            command = [True, False, False, False]

        key_list = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]
        keys = {key: command[i] for i, key in enumerate(key_list)}
        return keys


