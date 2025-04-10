import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import random
import os
import time
from collections import deque
import json

# Disable TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class GomokuGame:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1  # 1 for 'X', 2 for 'O'
        self.game_over = False
        self.winner = None
        self.move_history = []  # Track all moves for analysis
        
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history = []
        return self.get_state()
    
    def get_state(self):
        # Returns game state as 3D tensor: [board_size, board_size, 3]
        # Third channel represents whose turn it is
        state = np.zeros((self.board_size, self.board_size, 3), dtype=np.float32)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 1:  # X
                    state[i, j, 0] = 1
                elif self.board[i, j] == 2:  # O
                    state[i, j, 1] = 1
        
        # Set the third channel to indicate current player
        state[:, :, 2] = self.current_player - 1
        return state
    
    def get_valid_moves(self):
        return [(i, j) for i in range(self.board_size) for j in range(self.board_size) if self.board[i, j] == 0]
    
    def make_move(self, row, col):
        if self.game_over or row < 0 or row >= self.board_size or col < 0 or col >= self.board_size or self.board[row, col] != 0:
            return False
        
        self.board[row, col] = self.current_player
        self.move_history.append((row, col, self.current_player))
        
        # Check for win
        if self.check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        # Check for draw
        elif len(self.get_valid_moves()) == 0:
            self.game_over = True
            self.winner = 0  # Draw
        else:
            # Switch player
            self.current_player = 3 - self.current_player  # 1 -> 2, 2 -> 1
            
        return True
    
    def check_win(self, row, col):
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, diagonal, anti-diagonal
        
        for dr, dc in directions:
            count = 1
            # Check in positive direction
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            
            # Check in negative direction
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            
            if count >= 5:
                return True
        
        return False
    
    def print_board(self):
        symbols = {0: '.', 1: 'X', 2: 'O'}
    
        # Drukuj nagłówek kolumn
        print("    " + " ".join(f"{i:2d}" for i in range(self.board_size)))
        print("   " + "---" * self.board_size)
        
        # Drukuj wiersze planszy
        for i in range(self.board_size):
            # Formatuj numer wiersza, aby był wyrównany do prawej w polu o szerokości 2
            print(f"{i:2d} |", end=" ")
            
            # Drukuj zawartość wiersza
            for j in range(self.board_size):
                # Używaj stałej szerokości dla każdego symbolu
                print(f"{symbols[self.board[i, j]]} ", end=" ")
            
            print()  # Nowa linia na końcu wiersza
    
  
    def get_last_move(self):
        if self.move_history:
            return self.move_history[-1]
        return None


class GomokuAI:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.model = self._build_model()
        self.opponent_model = self._build_opponent_prediction_model()
        self.epsilon = 0.5  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.97
        self.memory = deque(maxlen=500000)  # Increased experience memory
        self.opponent_memory = deque(maxlen=250000)  # Memory for opponent move prediction
        self.win_patterns = []  # Store winning patterns
        self.threat_patterns = []  # Store threat patterns
        
    def _build_model(self):
        # Zwiększ złożoność modelu
        pattern_input = Input(shape=(7, 7, 3))
        
        # Dodaj więcej filtrów w warstwach konwolucyjnych
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001))(pattern_input)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Dodaj dodatkową warstwę konwolucyjną
        x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Zwiększ liczbę neuronów w warstwach gęstych
        flat = Flatten()(x)
        x = Dense(512, kernel_regularizer=l2(0.001))(flat)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.3)(x)
        
        x = Dense(256, kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.2)(x)
        
        output = Dense(1, activation='tanh')(x)
        
        model = Model(inputs=pattern_input, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')  # Zmniejsz learning rate
        
        return model

    
    def _build_opponent_prediction_model(self):
        # Input: current board state
        board_input = Input(shape=(self.board_size, self.board_size, 3))
        
        # Convolutional layers
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.001))(board_input)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Output layer - probability distribution over all possible moves
        x = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
        output = Flatten()(x)  # Flatten to get probabilities for each position
        
        # Build model
        model = Model(inputs=board_input, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
        
        return model
    
    # Dodaj nową metodę do klasy GomokuAI
    def remember_move_sequence(self, game_history):
        """Zapamiętuje sekwencje ruchów z całej gry."""
        if not hasattr(self, 'move_sequences'):
            self.move_sequences = []
        
        # Zapisz sekwencję ruchów jeśli nie jest za długa
        if len(game_history) > 5 and len(game_history) < 50:
            self.move_sequences.append(game_history)
            # Ogranicz liczbę zapamiętanych sekwencji
            if len(self.move_sequences) > 500:
                # Usuń losową starszą sekwencję
                del self.move_sequences[random.randint(0, len(self.move_sequences) - 101)]


    def extract_patterns(self, board, player):
        """Extracts all possible 7x7 patterns from the board."""
        opponent = 3 - player
        patterns = []
        positions = []
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == 0:  # only empty fields
                    # Extract 7x7 pattern centered at (i,j)
                    pattern = np.zeros((7, 7, 3))
                    for r in range(-3, 4):
                        for c in range(-3, 4):
                            if 0 <= i+r < self.board_size and 0 <= j+c < self.board_size:
                                if board[i+r, j+c] == player:
                                    pattern[r+3, c+3, 0] = 1  # player channel
                                elif board[i+r, j+c] == opponent:
                                    pattern[r+3, c+3, 1] = 1  # opponent channel
                    
                    # Set current player indicator in third channel
                    pattern[:, :, 2] = player - 1
                    
                    patterns.append(pattern)
                    positions.append((i, j))
        
        return np.array(patterns), positions
    
    def get_move(self, game, valid_moves):
        """Chooses the best move based on pattern evaluation or randomly."""
        if not valid_moves:  # Check if valid_moves is empty
            return None
            
        if random.random() < self.epsilon:
            # Exploration: random move
            return random.choice(valid_moves)
        else:
            # Exploitation: best move according to model
            best_move = self.get_best_move(game)
            if best_move in valid_moves:
                return best_move
            else:
                # Fallback to random move if best_move is not valid
                return random.choice(valid_moves)

    
    def get_best_move(self, game):
        """Zwraca najlepszy ruch na podstawie oceny wzorców i reguł strategicznych."""
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        
        # Sprawdź czy mamy podobną sekwencję ruchów w pamięci
        if hasattr(self, 'move_sequences') and len(game.move_history) >= 5:
            current_sequence = [move[:2] for move in game.move_history[-5:]]  # Ostatnie 5 ruchów
            
            best_continuation = None
            best_similarity = 0
            
            for sequence in self.move_sequences:
                # Znajdź podobne sekwencje
                for i in range(len(sequence) - len(current_sequence)):
                    seq_part = [move[:2] for move in sequence[i:i+len(current_sequence)]]
                    similarity = sum(1 for a, b in zip(current_sequence, seq_part) if a == b)
                    
                    if similarity >= len(current_sequence) * 0.8:  # 80% podobieństwa
                        # Znaleziono podobną sekwencję, sprawdź następny ruch
                        if i + len(current_sequence) < len(sequence):
                            next_move = sequence[i + len(current_sequence)][:2]
                            if next_move in valid_moves and similarity > best_similarity:
                                best_continuation = next_move
                                best_similarity = similarity
            
            if best_continuation:
                print(f"Using remembered sequence continuation: {best_continuation}")
                return best_continuation
        
        # Jeśli nie znaleziono kontynuacji sekwencji, użyj standardowej logiki
        player = game.current_player
        board = game.board.copy()
        
        # Move categorization
        critical_moves = []
        defensive_moves = []
        offensive_moves = []
        neutral_moves = []
        
        # Predict opponent's next move
        opponent_move_probs = self.predict_opponent_move(game)
        
        for move in valid_moves:
            row, col = move
            rule_score = evaluate_board_with_rules(board, player, move)
            
            # Add bonus for moves that counter predicted opponent moves
            if opponent_move_probs is not None:
                opponent_idx = row * self.board_size + col
                if opponent_idx < len(opponent_move_probs):
                    opponent_prob = opponent_move_probs[opponent_idx]
                    # If opponent is likely to play here, consider blocking
                    rule_score += opponent_prob * 5000
            
            # Categorize moves based on their score
            if rule_score >= 90000:  # Critical moves (win or block win)
                critical_moves.append((move, rule_score))
            elif rule_score >= 9000:  # Defensive moves (block fours, create fours)
                defensive_moves.append((move, rule_score))
            elif rule_score >= 900:   # Offensive moves (create threes, block threes)
                offensive_moves.append((move, rule_score))
            else:                     # Neutral moves
                neutral_moves.append((move, rule_score))
        
        # If we have critical moves, choose the best one
        if critical_moves:
            return max(critical_moves, key=lambda x: x[1])[0]
        
        # Choose which moves to evaluate with the neural network
        if defensive_moves:
            moves_to_evaluate = [move for move, _ in defensive_moves]
            best_move_category = defensive_moves
        elif offensive_moves:
            moves_to_evaluate = [move for move, _ in offensive_moves]
            best_move_category = offensive_moves
        else:
            moves_to_evaluate = [move for move, _ in neutral_moves]
            best_move_category = neutral_moves
        
        # If no moves to evaluate, choose a random move
        if not moves_to_evaluate:
            return random.choice(valid_moves)
        
        # Use neural network to evaluate selected moves
        patterns = []
        positions = []
        
        for move in moves_to_evaluate:
            row, col = move
            # Temporarily make the move
            temp_board = board.copy()
            temp_board[row, col] = player
            # Extract pattern
            pattern, _ = self.extract_patterns(temp_board, player)
            if len(pattern) > 0:
                patterns.append(pattern[0])
                positions.append(move)
        
        if not patterns:
            # If no patterns could be generated, choose the best move based on rules
            if best_move_category:
                return max(best_move_category, key=lambda x: x[1])[0]
            else:
                return random.choice(valid_moves)
        
        # Evaluate patterns using the network
        patterns_array = np.array(patterns)
        evaluations = self.model.predict(patterns_array, verbose=0).flatten()
        
        # Combine network evaluations with rule-based scores
        combined_scores = []
        for i, move in enumerate(positions):
            rule_score = next((score for m, score in best_move_category if m == move), 0)
            neural_score = evaluations[i]
            
            # Dynamic weighting - higher rule score gets more weight
            rule_weight = min(0.9, 0.5 + rule_score / 200000)
            neural_weight = 1.0 - rule_weight
            
            combined_score = rule_weight * rule_score + neural_weight * neural_score * 10000
            combined_scores.append((move, combined_score))
        
        # Choose the best move
        if combined_scores:
            return max(combined_scores, key=lambda x: x[1])[0]
        else:
            # Fallback - choose the best move based on rules
            if best_move_category:
                return max(best_move_category, key=lambda x: x[1])[0]
            else:
                return random.choice(valid_moves)


    def predict_opponent_move(self, game):
        """Predicts the opponent's next move."""
        if game.current_player == 1:
            opponent = 2
        else:
            opponent = 1
            
        # Get current state
        state = game.get_state()
                # Reshape for model input (add batch dimension)
        state_input = np.expand_dims(state, axis=0)
        
        # Predict opponent's move probabilities
        try:
            move_probs = self.opponent_model.predict(state_input, verbose=0).flatten()
            
            # Mask out invalid moves
            valid_moves_mask = np.zeros(self.board_size * self.board_size)
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if game.board[i, j] == 0:  # Empty cell
                        valid_moves_mask[i * self.board_size + j] = 1
            
            # Apply mask
            move_probs = move_probs * valid_moves_mask
            
            # Normalize probabilities
            if np.sum(move_probs) > 0:
                move_probs = move_probs / np.sum(move_probs)
            
            return move_probs
        except:
            # If prediction fails, return None
            return None
    
    def remember(self, pattern, reward):
        """Zapamiętuje wzorzec i jego ocenę do późniejszego treningu."""
        self.memory.append((pattern, reward))
        
        # Zapamiętuj więcej wzorców wygrywających z różnymi progami
        if reward > 0.8:
            self.win_patterns.append((pattern, reward))
            # Zwiększ limit najlepszych wzorców
            if len(self.win_patterns) > 2000:  # Zwiększ z 1000 na 2000
                self.win_patterns = sorted(self.win_patterns, key=lambda x: x[1], reverse=True)[:2000]
        
        # Dodaj zapamiętywanie wzorców defensywnych (blokujących)
        elif reward < -0.5:
            if not hasattr(self, 'defensive_patterns'):
                self.defensive_patterns = []
            self.defensive_patterns.append((pattern, reward))
            if len(self.defensive_patterns) > 1000:
                self.defensive_patterns = sorted(self.defensive_patterns, key=lambda x: x[1])[:1000]
    
    def remember_opponent_move(self, board_state, move):
        """Remembers opponent's move for prediction training."""
        row, col = move
        # Create target array (one-hot encoding of the move)
        target = np.zeros(self.board_size * self.board_size)
        target[row * self.board_size + col] = 1
        
        self.opponent_memory.append((board_state, target))
    
    def replay(self, batch_size=256, epochs=5):  # Zwiększ batch_size
        """Trenuje model na podstawie zapamiętanych doświadczeń."""
        if len(self.memory) < batch_size:
            return
        
        # Wybierz próbkę z pamięci
        minibatch = random.sample(self.memory, batch_size)
        
        # Dodaj wzorce wygrywające
        if self.win_patterns:
            win_samples = random.sample(self.win_patterns, min(len(self.win_patterns), batch_size // 3))
            minibatch = minibatch[:batch_size - len(win_samples)] + win_samples
        
        # Dodaj wzorce defensywne jeśli istnieją
        if hasattr(self, 'defensive_patterns') and self.defensive_patterns:
            defensive_samples = random.sample(self.defensive_patterns, 
                                            min(len(self.defensive_patterns), batch_size // 6))
            minibatch = minibatch[:batch_size - len(win_samples) - len(defensive_samples)] + win_samples + defensive_samples
        
        patterns = np.array([item[0] for item in minibatch])
        rewards = np.array([item[1] for item in minibatch])
        
        # Dodaj augmentację danych - obroty i odbicia wzorców
        if random.random() < 0.5:  # 50% szans na augmentację
            augmented_patterns = []
            augmented_rewards = []
            
            for i, pattern in enumerate(patterns):
                # Dodaj oryginalny wzorzec
                augmented_patterns.append(pattern)
                augmented_rewards.append(rewards[i])
                
                # Dodaj obrócony wzorzec (90 stopni)
                rotated = np.rot90(pattern, k=1, axes=(0, 1))
                augmented_patterns.append(rotated)
                augmented_rewards.append(rewards[i])
            
            patterns = np.array(augmented_patterns)
            rewards = np.array(augmented_rewards)
        
        # Trenuj model
        history = self.model.fit(patterns, rewards, epochs=epochs, 
                                batch_size=min(128, len(patterns)), verbose=1)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        print(f"Model trained on {len(patterns)} patterns. Loss: {history.history['loss'][-1]:.4f}")
        
        # Trenuj model przewidywania ruchów przeciwnika
        self.train_opponent_prediction()
        
        return history

    
    def train_opponent_prediction(self, batch_size=128, epochs=3):
        """Trains the opponent prediction model."""
        if len(self.opponent_memory) < batch_size:
            return
        
        # Randomly select a sample from memory
        minibatch = random.sample(self.opponent_memory, batch_size)
        states = np.array([item[0] for item in minibatch])
        targets = np.array([item[1] for item in minibatch])
        
        # Train model
        history = self.opponent_model.fit(states, targets, epochs=epochs, batch_size=min(64, batch_size), verbose=0)
        
        print(f"Opponent prediction model trained. Loss: {history.history['loss'][-1]:.4f}")
        
        return history
    
    def train_on_patterns(self, patterns, labels, epochs=10, batch_size=64):
        """Trains the network based on patterns and their evaluations."""
        history = self.model.fit(patterns, labels, epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return history
    
    def save_model(self, filepath):
        """Saves the model to a file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        
        # Save opponent prediction model
        opponent_filepath = filepath.replace('.h5', '_opponent.h5')
        self.opponent_model.save(opponent_filepath)
        
        # Save winning patterns
        if self.win_patterns:
            patterns_filepath = filepath.replace('.h5', '_win_patterns.npz')
            win_patterns_data = np.array([p[0] for p in self.win_patterns])
            win_patterns_rewards = np.array([p[1] for p in self.win_patterns])
            np.savez(patterns_filepath, patterns=win_patterns_data, rewards=win_patterns_rewards)
        
        print(f"Model saved to {filepath}")
        print(f"Opponent model saved to {opponent_filepath}")
    
    def load_model(self, filepath):
        """Loads the model from a file."""
        try:
            self.model = load_model(filepath)
            print(f"Model loaded from {filepath}")
            
            # Try to load opponent prediction model
            opponent_filepath = filepath.replace('.h5', '_opponent.h5')
            if os.path.exists(opponent_filepath):
                self.opponent_model = load_model(opponent_filepath)
                print(f"Opponent model loaded from {opponent_filepath}")
            
            # Try to load winning patterns
            patterns_filepath = filepath.replace('.h5', '_win_patterns.npz')
            if os.path.exists(patterns_filepath):
                data = np.load(patterns_filepath)
                patterns = data['patterns']
                rewards = data['rewards']
                self.win_patterns = [(patterns[i], rewards[i]) for i in range(len(rewards))]
                print(f"Loaded {len(self.win_patterns)} winning patterns")
                
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


def evaluate_board_with_rules(board, player, position):
    """Evaluates a move based on strategic Gomoku rules."""
    board_copy = board.copy()
    row, col = position
    opponent = 3 - player
    score = 0
    
    # Temporarily make the move
    board_copy[row, col] = player
    
    # Check if this move gives a win (5 in a row)
    if check_win_at_position(board_copy, row, col):
        score += 100000  # Highest priority for winning
    
    # Check if it blocks opponent's win
    board_copy[row, col] = opponent
    if check_win_at_position(board_copy, row, col):
        score += 95000  # Very high priority for blocking a win
    
    # Restore board to state with our move
    board_copy[row, col] = player
    
    # Check for open fours (4 in a row with both ends open)
    open_fours = count_open_sequences(board_copy, player, row, col, 4)
    if open_fours > 0:
        score += 20000 * open_fours  # High priority for open fours
    
    # Check for semi-open fours (4 in a row with one end open)
    semi_open_fours = count_semi_open_sequences(board_copy, player, row, col, 4)
    if semi_open_fours > 0:
        score += 10000 * semi_open_fours
    
    # Check for blocking opponent's open fours
    board_copy[row, col] = opponent
    opponent_open_fours = count_open_sequences(board_copy, opponent, row, col, 4)
    if opponent_open_fours > 0:
        score += 19000 * opponent_open_fours  # High priority for blocking fours
    
    # Check for blocking opponent's semi-open fours
    opponent_semi_open_fours = count_semi_open_sequences(board_copy, opponent, row, col, 4)
    if opponent_semi_open_fours > 0:
        score += 9500 * opponent_semi_open_fours
    
    # Restore board to state with our move
    board_copy[row, col] = player
    
    # Check for double open threes (creates two open threes)
    double_open_threes = count_double_open_threes(board_copy, player, row, col)
    if double_open_threes > 0:
        score += 9000 * double_open_threes  # Very high priority for double threat
    
    # Check for open threes (3 in a row with both ends open)
    open_threes = count_open_sequences(board_copy, player, row, col, 3)
    if open_threes > 0:
        score += 5000 * open_threes  # Medium priority for open threes
    
    # Check for semi-open threes
    semi_open_threes = count_semi_open_sequences(board_copy, player, row, col, 3)
    if semi_open_threes > 0:
        score += 1000 * semi_open_threes
    
    # Check for blocking opponent's double open threes
    board_copy[row, col] = opponent
    opponent_double_open_threes = count_double_open_threes(board_copy, opponent, row, col)
    if opponent_double_open_threes > 0:
        score += 8500 * opponent_double_open_threes
    
    # Check for blocking opponent's open threes
    opponent_open_threes = count_open_sequences(board_copy, opponent, row, col, 3)
    if opponent_open_threes > 0:
        score += 4500 * opponent_open_threes  # Medium priority for blocking threes
    
    # Check for blocking opponent's semi-open threes
    opponent_semi_open_threes = count_semi_open_sequences(board_copy, opponent, row, col, 3)
    if opponent_semi_open_threes > 0:
        score += 900 * opponent_semi_open_threes
    
    # Restore board to state with our move
    board_copy[row, col] = player
    
    # Check for open twos
    open_twos = count_open_sequences(board_copy, player, row, col, 2)
    if open_twos > 0:
        score += 500 * open_twos  # Low priority for open twos
    
    # Check for blocking opponent's open twos
    board_copy[row, col] = opponent
    opponent_open_twos = count_open_sequences(board_copy, opponent, row, col, 2)
    if opponent_open_twos > 0:
        score += 400 * opponent_open_twos  # Low priority for blocking twos
    
    # Restore board
    board_copy[row, col] = player
    
    # Additional points for move in center area
    board_size = len(board)
    center = board_size // 2
    distance_to_center = abs(row - center) + abs(col - center)
    
    # More sophisticated center evaluation - higher priority for early game
    empty_count = np.sum(board == 0)
    total_cells = board_size * board_size
    game_progress = 1 - (empty_count / total_cells)  # 0 at start, 1 at end
    
    # Center importance decreases as game progresses
    center_importance = max(0, 1 - game_progress * 2)  # Linearly decreases to 0 at 50% filled
    center_score = max(0, 300 - 30 * distance_to_center) * center_importance
    score += center_score
    
    # Bonus for moves that extend existing lines
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        extension_bonus = check_line_extension(board, player, row, col, dr, dc)
        score += extension_bonus * 50
    
    return score


def check_win_at_position(board, row, col):
    """Checks if a move at position (row, col) gives a win."""
    player = board[row, col]
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, diagonal
    
    for dr, dc in directions:
        count = 1  # Counter for pieces in line
        
        # Check in one direction
        r, c = row + dr, col + dc
        while 0 <= r < len(board) and 0 <= c < len(board) and board[r, c] == player:
            count += 1
            r += dr
            c += dc
        
        # Check in opposite direction
        r, c = row - dr, col - dc
        while 0 <= r < len(board) and 0 <= c < len(board) and board[r, c] == player:
            count += 1
            r -= dr
            c -= dc
        
        if count >= 5:
            return True
    
    return False


def count_open_sequences(board, player, row, col, length):
    """Counts open sequences (with empty cells at both ends) of given length."""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    open_sequences = 0
    
    for dr, dc in directions:
        # Check sequence in this direction
        sequence_length = 1  # Start with 1 (current position)
        open_ends = 0  # Counter for open ends
        
        # Check in one direction
        r, c = row + dr, col + dc
        while 0 <= r < len(board) and 0 <= c < len(board) and board[r, c] == player:
            sequence_length += 1
            r += dr
            c += dc
        
        # Check if end is open
        if 0 <= r < len(board) and 0 <= c < len(board) and board[r, c] == 0:
            open_ends += 1
        
        # Check in opposite direction
        r, c = row - dr, col - dc
        while 0 <= r < len(board) and 0 <= c < len(board) and board[r, c] == player:
            sequence_length += 1
            r -= dr
            c -= dc
        
        # Check if second end is open
        if 0 <= r < len(board) and 0 <= c < len(board) and board[r, c] == 0:
            open_ends += 1
        
        # If sequence has correct length and is open at both ends
        if sequence_length == length and open_ends == 2:
            open_sequences += 1
    
    return open_sequences


def count_semi_open_sequences(board, player, row, col, length):
    """Counts semi-open sequences (with empty cell at only one end) of given length."""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    semi_open_sequences = 0
    
    for dr, dc in directions:
        # Check sequence in this direction
        sequence_length = 1  # Start with 1 (current position)
        open_ends = 0  # Counter for open ends
        
        # Check in one direction
        r, c = row + dr, col + dc
        while 0 <= r < len(board) and 0 <= c < len(board) and board[r, c] == player:
            sequence_length += 1
            r += dr
            c += dc
        
        # Check if end is open
        if 0 <= r < len(board) and 0 <= c < len(board) and board[r, c] == 0:
            open_ends += 1
        
        # Check in opposite direction
        r, c = row - dr, col - dc
        while 0 <= r < len(board) and 0 <= c < len(board) and board[r, c] == player:
            sequence_length += 1
            r -= dr
            c -= dc
        
        # Check if second end is open
        if 0 <= r < len(board) and 0 <= c < len(board) and board[r, c] == 0:
            open_ends += 1
        
        # If sequence has correct length and is open at exactly one end
        if sequence_length == length and open_ends == 1:
            semi_open_sequences += 1
    
    return semi_open_sequences


def count_double_open_threes(board, player, row, col):
    """Counts how many pairs of open threes are created by this move."""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    open_three_directions = []
    
    for dr, dc in directions:
        # Check for open three in this direction
        sequence_length = 1
        open_ends = 0
        
        # Check in one direction
        r, c = row + dr, col + dc
        while 0 <= r < len(board) and 0 <= c < len(board) and board[r, c] == player:
            sequence_length += 1
            r += dr
            c += dc
        
        # Check if end is open
        if 0 <= r < len(board) and 0 <= c < len(board) and board[r, c] == 0:
            open_ends += 1
        
        # Check in opposite direction
        r, c = row - dr, col - dc
        while 0 <= r < len(board) and 0 <= c < len(board) and board[r, c] == player:
            sequence_length += 1
            r -= dr
            c -= dc
        
        # Check if second end is open
        if 0 <= r < len(board) and 0 <= c < len(board) and board[r, c] == 0:
            open_ends += 1
        
        # If this forms an open three
        if sequence_length == 3 and open_ends == 2:
            open_three_directions.append((dr, dc))
    
    # Count pairs of open threes (double threats)
    return len(open_three_directions) // 2


def check_line_extension(board, player, row, col, dr, dc):
    """Checks if this move extends an existing line and returns a bonus score."""
    # Check if there are any player pieces in this direction
    has_extension = False
    
    # Check in one direction
    r, c = row + dr, col + dc
    while 0 <= r < len(board) and 0 <= c < len(board):
        if board[r, c] == player:
            has_extension = True
            break
        elif board[r, c] != 0:  # Opponent piece
            break
        r += dr
        c += dc
    
    if not has_extension:
        # Check in opposite direction
        r, c = row - dr, col - dc
        while 0 <= r < len(board) and 0 <= c < len(board):
            if board[r, c] == player:
                has_extension = True
                break
            elif board[r, c] != 0:  # Opponent piece
                break
            r -= dr
            c -= dc
    
    return 1 if has_extension else 0


def generate_strategic_patterns(num_patterns=100000, board_size=15):
    """Generates strategic training patterns for the network."""
    patterns = []
    labels = []
    
    # 1. Winning patterns (5 in a row)
    for _ in range(num_patterns // 10):
        pattern = np.zeros((7, 7, 3))
        player_channel = 0
        
        # Place 5 pieces in a row
        direction = random.choice([(1, 0), (0, 1), (1, 1), (1, -1)])
        dr, dc = direction
        
        # Safe range calculation
        max_r = max(0, 6 - 4*abs(dr))
        max_c = max(0, 6 - 4*abs(dc))
        
        if max_r > 0:
            start_r = random.randint(0, max_r)
        else:
            start_r = 0
            
        if max_c > 0:
            start_c = random.randint(0, max_c)
        else:
            start_c = 0
        
        for i in range(5):
            if 0 <= start_r + i*dr < 7 and 0 <= start_c + i*dc < 7:
                pattern[start_r + i*dr, start_c + i*dc, player_channel] = 1
        
        # Set current player indicator
        pattern[:, :, 2] = 0  # Player 1
        
        patterns.append(pattern)
        labels.append(1.0)  # Highest evaluation
    
    # 2. Open four patterns
    for _ in range(num_patterns // 10):
        pattern = np.zeros((7, 7, 3))
        player_channel = 0
        
        # Place 4 pieces in a row with empty cells at ends
        direction = random.choice([(1, 0), (0, 1), (1, 1), (1, -1)])
        dr, dc = direction
        
        # Safe range calculation
        max_r = max(0, 6 - 5*abs(dr))
        max_c = max(0, 6 - 5*abs(dc))
        
        if max_r > 0:
            start_r = random.randint(0, max_r)
        else:
            start_r = 0
            
        if max_c > 0:
            start_c = random.randint(0, max_c)
        else:
            start_c = 0
        
        # Ensure both ends are empty
        for i in range(1, 5):
            if 0 <= start_r + i*dr < 7 and 0 <= start_c + i*dc < 7:
                pattern[start_r + i*dr, start_c + i*dc, player_channel] = 1
        
        # Set current player indicator
        pattern[:, :, 2] = 0  # Player 1
        
        patterns.append(pattern)
        labels.append(0.9)  # Very high evaluation
    
    # 3. Double open three patterns
    for _ in range(num_patterns // 10):
        pattern = np.zeros((7, 7, 3))
        player_channel = 0
        
        # Create first open three
        direction1 = random.choice([(1, 0), (0, 1), (1, 1), (1, -1)])
        dr1, dc1 = direction1
        
        # Choose a different direction for second open three
        available_directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        available_directions.remove(direction1)
        direction2 = random.choice(available_directions)
        dr2, dc2 = direction2
        
        # Place center piece (shared by both lines)
        center_r, center_c = 3, 3
        pattern[center_r, center_c, player_channel] = 1
        
        # Place first open three
        for i in range(1, 3):
            r, c = center_r + i*dr1, center_c + i*dc1
            if 0 <= r < 7 and 0 <= c < 7:
                pattern[r, c, player_channel] = 1
        
        # Place second open three
        for i in range(1, 3):
            r, c = center_r + i*dr2, center_c + i*dc2
            if 0 <= r < 7 and 0 <= c < 7:
                pattern[r, c, player_channel] = 1
        
        # Set current player indicator
        pattern[:, :, 2] = 0  # Player 1
        
        patterns.append(pattern)
        labels.append(0.85)  # Very high evaluation
    
    # 4. Open three patterns
    for _ in range(num_patterns // 10):
        pattern = np.zeros((7, 7, 3))
        player_channel = 0
        
        direction = random.choice([(1, 0), (0, 1), (1, 1), (1, -1)])
        dr, dc = direction
        
        # Safe range calculation
        max_r = max(1, 5 - 2*abs(dr))
        max_c = max(1, 5 - 2*abs(dc))
        
        if max_r >= 1:
            start_r = random.randint(1, max_r)
        else:
            start_r = 1
            
        if max_c >= 1:
            start_c = random.randint(1, max_c)
        else:
            start_c = 1
        
        # Place 3 pieces in a row with empty cells at ends
        for i in range(3):
            if 0 <= start_r + i*dr < 7 and 0 <= start_c + i*dc < 7:
                pattern[start_r + i*dr, start_c + i*dc, player_channel] = 1
        
        # Set current player indicator
        pattern[:, :, 2] = 0  # Player 1
        
        patterns.append(pattern)
        labels.append(0.7)  # High evaluation
    
    # 5. Blocking opponent's win patterns
    for _ in range(num_patterns // 10):
        pattern = np.zeros((7, 7, 3))
        opponent_channel = 1
        
        # Opponent has 4 in a row
        direction = random.choice([(1, 0), (0, 1), (1, 1), (1, -1)])
        dr, dc = direction
        
        # Safe range calculation
        max_r = max(0, 6 - 4*abs(dr))
        max_c = max(0, 6 - 4*abs(dc))
        
        if max_r > 0:
            start_r = random.randint(0, max_r)
        else:
            start_r = 0
            
        if max_c > 0:
            start_c = random.randint(0, max_c)
        else:
            start_c = 0
        
        for i in range(4):
            if 0 <= start_r + i*dr < 7 and 0 <= start_c + i*dc < 7:
                pattern[start_r + i*dr, start_c + i*dc, opponent_channel] = 1
        
        # Set current player indicator
        pattern[:, :, 2] = 0  # Player 1
        
        patterns.append(pattern)
        labels.append(0.8)  # Very high evaluation (blocking win)
    
    # 6. Blocking opponent's open four patterns
    for _ in range(num_patterns // 10):
        pattern = np.zeros((7, 7, 3))
        opponent_channel = 1
        
        # Opponent has open four
        direction = random.choice([(1, 0), (0, 1), (1, 1), (1, -1)])
        dr, dc = direction
        
        # Safe range calculation
        max_r = max(1, 5 - 3*abs(dr))
        max_c = max(1, 5 - 3*abs(dc))
        
        if max_r >= 1:
            start_r = random.randint(1, max_r)
        else:
            start_r = 1
            
        if max_c >= 1:
            start_c = random.randint(1, max_c)
        else:
            start_c = 1
        
        # Place opponent's 3 pieces with one empty cell in the middle
        pattern[start_r, start_c, opponent_channel] = 1
        pattern[start_r + 2*dr, start_c + 2*dc, opponent_channel] = 1
        pattern[start_r + 3*dr, start_c + 3*dc, opponent_channel] = 1
        
        # Set current player indicator
        pattern[:, :, 2] = 0  # Player 1
        
        patterns.append(pattern)
        labels.append(0.75)  # High evaluation (blocking potential four)
    
    # 7. Blocking opponent's double open three patterns
    for _ in range(num_patterns // 10):
        pattern = np.zeros((7, 7, 3))
        opponent_channel = 1
        
        # Create opponent's first open three
        direction1 = random.choice([(1, 0), (0, 1), (1, 1), (1, -1)])
        dr1, dc1 = direction1
        
        # Choose a different direction for second open three
        available_directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        available_directions.remove(direction1)
        direction2 = random.choice(available_directions)
        dr2, dc2 = direction2
        
        # Place center piece (shared by both lines)
        center_r, center_c = 3, 3
        pattern[center_r, center_c, opponent_channel] = 1
        
        # Place first open three
        for i in range(1, 3):
            r, c = center_r + i*dr1, center_c + i*dc1
            if 0 <= r < 7 and 0 <= c < 7:
                pattern[r, c, opponent_channel] = 1
        
        # Place second open three
        for i in range(1, 3):
            r, c = center_r + i*dr2, center_c + i*dc2
            if 0 <= r < 7 and 0 <= c < 7:
                pattern[r, c, opponent_channel] = 1
        
        # Set current player indicator
        pattern[:, :, 2] = 0  # Player 1
        
        patterns.append(pattern)
        labels.append(0.7)  # High evaluation (blocking double threat)
    
        # 8. Neutral patterns with some strategic value
    for _ in range(num_patterns // 10):
        pattern = np.zeros((7, 7, 3))
        player_channel = 0
        opponent_channel = 1
        
        # Place 2-3 player pieces in strategic positions
        num_player_pieces = random.randint(2, 3)
        for _ in range(num_player_pieces):
            r, c = random.randint(1, 5), random.randint(1, 5)
            if pattern[r, c, 0] == 0 and pattern[r, c, 1] == 0:
                pattern[r, c, player_channel] = 1
        
        # Place 1-2 opponent pieces
        num_opponent_pieces = random.randint(1, 2)
        for _ in range(num_opponent_pieces):
            r, c = random.randint(1, 5), random.randint(1, 5)
            if pattern[r, c, 0] == 0 and pattern[r, c, 1] == 0:
                pattern[r, c, opponent_channel] = 1
        
        # Set current player indicator
        pattern[:, :, 2] = 0  # Player 1
        
        patterns.append(pattern)
        labels.append(0.4)  # Medium evaluation
    
    # 9. Defensive patterns
    for _ in range(num_patterns // 10):
        pattern = np.zeros((7, 7, 3))
        player_channel = 0
        opponent_channel = 1
        
        # Place opponent pieces in threatening position
        direction = random.choice([(1, 0), (0, 1), (1, 1), (1, -1)])
        dr, dc = direction
        
        # Place 2-3 opponent pieces in a row
        start_r, start_c = random.randint(1, 4), random.randint(1, 4)
        for i in range(random.randint(2, 3)):
            r, c = start_r + i*dr, start_c + i*dc
            if 0 <= r < 7 and 0 <= c < 7:
                pattern[r, c, opponent_channel] = 1
        
        # Place 1-2 player pieces nearby for defense
        for _ in range(random.randint(1, 2)):
            offset_r = random.randint(-1, 1)
            offset_c = random.randint(-1, 1)
            r, c = start_r + offset_r, start_c + offset_c
            if 0 <= r < 7 and 0 <= c < 7 and pattern[r, c, 0] == 0 and pattern[r, c, 1] == 0:
                pattern[r, c, player_channel] = 1
        
        # Set current player indicator
        pattern[:, :, 2] = 0  # Player 1
        
        patterns.append(pattern)
        labels.append(0.5)  # Medium-high evaluation (defensive value)
    
    # 10. Low value patterns
    for _ in range(num_patterns // 10):
        pattern = np.zeros((7, 7, 3))
        player_channel = 0
        opponent_channel = 1
        
        # Randomly place pieces with no clear strategic value
        for _ in range(random.randint(3, 8)):
            r, c = random.randint(0, 6), random.randint(0, 6)
            channel = random.randint(0, 1)
            if pattern[r, c, 0] == 0 and pattern[r, c, 1] == 0:
                pattern[r, c, channel] = 1
        
        # Set current player indicator
        pattern[:, :, 2] = 0  # Player 1
        
        patterns.append(pattern)
        labels.append(0.1)  # Low evaluation
    
    return np.array(patterns), np.array(labels)


def generate_expert_patterns(num_patterns=5000):
    """Generates patterns from expert games and curated examples."""
    patterns = []
    labels = []
    
    # 1. Expert winning moves
    for _ in range(num_patterns // 5):
        pattern = np.zeros((7, 7, 3))
        player_channel = 0
        
        # Create a winning pattern with a specific shape
        # Example: Create a "hook" shape (strong tactical pattern)
        center_r, center_c = 3, 3
        
        # Place the center piece
        pattern[center_r, center_c, player_channel] = 1
        
        # Create horizontal line
        for i in range(1, 3):
            pattern[center_r, center_c + i, player_channel] = 1
        
        # Create vertical line
        for i in range(1, 3):
            pattern[center_r + i, center_c, player_channel] = 1
        
        # Set current player indicator
        pattern[:, :, 2] = 0  # Player 1
        
        patterns.append(pattern)
        labels.append(0.95)  # Very high evaluation
    
    # 2. Expert defensive moves
    for _ in range(num_patterns // 5):
        pattern = np.zeros((7, 7, 3))
        player_channel = 0
        opponent_channel = 1
        
        # Create a pattern where opponent has a strong position but there's a key defensive move
        # Example: Opponent has a "hook" shape, we block the critical point
        center_r, center_c = 3, 3
        
        # Place opponent pieces
        pattern[center_r-1, center_c, opponent_channel] = 1
        pattern[center_r-1, center_c+1, opponent_channel] = 1
        pattern[center_r, center_c+1, opponent_channel] = 1
        pattern[center_r+1, center_c+1, opponent_channel] = 1
        
        # Place our defensive piece
        pattern[center_r, center_c, player_channel] = 1
        
        # Set current player indicator
        pattern[:, :, 2] = 0  # Player 1
        
        patterns.append(pattern)
        labels.append(0.85)  # High evaluation
    
    # 3. Expert opening moves
    for _ in range(num_patterns // 5):
        pattern = np.zeros((7, 7, 3))
        player_channel = 0
        
        # Create early game patterns
        # Example: Place pieces near the center with good spacing
        center_r, center_c = 3, 3
        
        # Place 2-3 pieces in good positions
        pattern[center_r, center_c, player_channel] = 1
        
        # Add 1-2 more pieces in strategic positions
        offsets = [(2, 0), (0, 2), (2, 2), (-2, 0), (0, -2), (-2, -2)]
        selected_offsets = random.sample(offsets, random.randint(1, 2))
        
        for dr, dc in selected_offsets:
            r, c = center_r + dr, center_c + dc
            if 0 <= r < 7 and 0 <= c < 7:
                pattern[r, c, player_channel] = 1
        
        # Set current player indicator
        pattern[:, :, 2] = 0  # Player 1
        
        patterns.append(pattern)
        labels.append(0.7)  # High evaluation
    
    # 4. Expert counter-attack patterns
    for _ in range(num_patterns // 5):
        pattern = np.zeros((7, 7, 3))
        player_channel = 0
        opponent_channel = 1
        
        # Create a pattern where we counter-attack opponent's position
        center_r, center_c = 3, 3
        
        # Place opponent pieces
        for _ in range(3):
            r = random.randint(1, 5)
            c = random.randint(1, 5)
            if pattern[r, c, 0] == 0 and pattern[r, c, 1] == 0:
                pattern[r, c, opponent_channel] = 1
        
        # Place our counter-attack pieces
        pattern[center_r, center_c, player_channel] = 1
        
        # Add another piece that creates a threat
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        dr, dc = random.choice(directions)
        
        r, c = center_r + dr, center_c + dc
        if 0 <= r < 7 and 0 <= c < 7 and pattern[r, c, 0] == 0 and pattern[r, c, 1] == 0:
            pattern[r, c, player_channel] = 1
        
        # Set current player indicator
        pattern[:, :, 2] = 0  # Player 1
        
        patterns.append(pattern)
        labels.append(0.75)  # High evaluation
    
    # 5. Expert endgame patterns
    for _ in range(num_patterns // 5):
        pattern = np.zeros((7, 7, 3))
        player_channel = 0
        opponent_channel = 1
        
        # Create a crowded board with a key winning move
        # Fill about 60-70% of the board
        num_pieces = random.randint(25, 35)
        
        for _ in range(num_pieces):
            r = random.randint(0, 6)
            c = random.randint(0, 6)
            channel = random.randint(0, 1)
            if pattern[r, c, 0] == 0 and pattern[r, c, 1] == 0:
                pattern[r, c, channel] = 1
        
        # Place a key winning piece
        center_r, center_c = 3, 3
        if pattern[center_r, center_c, 0] == 0 and pattern[center_r, center_c, 1] == 0:
            pattern[center_r, center_c, player_channel] = 1
        
        # Set current player indicator
        pattern[:, :, 2] = 0  # Player 1
        
        patterns.append(pattern)
        labels.append(0.9)  # Very high evaluation
    
    return np.array(patterns), np.array(labels)

def generate_pro_game_patterns(num_patterns=5000):
    """Generuje wzorce na podstawie profesjonalnych gier Gomoku."""
    patterns = []
    labels = []
    
    # Tutaj możesz dodać kod do wczytywania prawdziwych gier profesjonalnych
    # z pliku lub bazy danych, ale na potrzeby przykładu generujemy syntetyczne wzorce
    
    # Wzorce otwarcia gry (pierwsze 5-10 ruchów)
    for _ in range(num_patterns // 4):
        pattern = np.zeros((7, 7, 3))
        player_channel = 0
        opponent_channel = 1
        
        # Symuluj popularne otwarcia
        # Przykład: wzorzec "gwiazda" - ruchy blisko środka
        center = 3
        moves = [(center, center)]
        
        # Dodaj 3-6 ruchów w popularnych pozycjach otwarcia
        num_moves = random.randint(3, 6)
        possible_offsets = [(1, 1), (1, -1), (-1, 1), (-1, -1), (2, 0), (0, 2), (-2, 0), (0, -2)]
        selected_offsets = random.sample(possible_offsets, num_moves)
        
        for dr, dc in selected_offsets:
            moves.append((center + dr, center + dc))
        
        # Umieść ruchy na planszy, naprzemiennie dla obu graczy
        for i, (r, c) in enumerate(moves):
            if 0 <= r < 7 and 0 <= c < 7:
                channel = player_channel if i % 2 == 0 else opponent_channel
                pattern[r, c, channel] = 1
        
        # Ustaw wskaźnik aktualnego gracza
        pattern[:, :, 2] = 0  # Player 1
        
        patterns.append(pattern)
        labels.append(0.7)  # Wysoka ocena dla profesjonalnych otwarć
    
    # [Dodaj więcej typów wzorców z profesjonalnych gier]
    
    return np.array(patterns), np.array(labels)


def train_initial_model(ai, epochs=10, num_patterns=10000, num_expert_patterns=5000):
    """Trenuje model na podstawie wygenerowanych wzorców."""
    print("Generating strategic training patterns...")
    try:
        patterns, labels = generate_strategic_patterns(num_patterns, ai.board_size)
        print(f"Successfully generated {len(patterns)} strategic patterns.")
        
        # Generuj wzorce eksperckie
        expert_patterns, expert_labels = generate_expert_patterns(num_expert_patterns)
        print(f"Successfully generated {len(expert_patterns)} expert patterns.")
        
        # Generuj wzorce z profesjonalnych gier
        pro_patterns, pro_labels = generate_pro_game_patterns(num_patterns // 2)
        print(f"Successfully generated {len(pro_patterns)} professional game patterns.")
        
        # Połącz wszystkie wzorce
        all_patterns = np.concatenate([patterns, expert_patterns, pro_patterns])
        all_labels = np.concatenate([labels, expert_labels, pro_labels])
        
        # Wymieszaj dane
        indices = np.arange(len(all_patterns))
        np.random.shuffle(indices)
        all_patterns = all_patterns[indices]
        all_labels = all_labels[indices]
        
        print(f"Training model on {len(all_patterns)} combined patterns...")
    except Exception as e:
        print(f"Error generating patterns: {e}")
        return False
    
    try:
        ai.train_on_patterns(all_patterns, all_labels, epochs=epochs)
        print("Initial training completed!")
        return True
    except Exception as e:
        print(f"Error during training: {e}")
        return False


# In the train_by_self_play function, ensure it always returns the ai object
def train_by_self_play(ai, num_games=20, board_size=15):
    """Trains model by playing games against itself."""
    game = GomokuGame(board_size)
    
    print(f"Starting self-play training for {num_games} games...")
    
    try:  # Add error handling
        for game_num in range(num_games):
            print(f"Game {game_num+1}/{num_games}")
            game.reset()
            game_patterns = []
            game_positions = []
            game_players = []
            game_states = []
            
            # First move near the center with some randomness
            mid = board_size // 2
            offset_r = random.randint(-2, 2)
            offset_c = random.randint(-2, 2)
            game.make_move(mid + offset_r, mid + offset_c)
            
            move_count = 0
            
            while not game.game_over and move_count < board_size * board_size:
                move_count += 1
                
                # Get patterns and positions
                player = game.current_player
                state = game.get_state()
                patterns, positions = ai.extract_patterns(game.board, player)
                
                if len(patterns) == 0:
                    break
                    
                # Save patterns, positions and state
                for i, pos in enumerate(positions):
                    game_patterns.append(patterns[i])
                    game_positions.append(pos)
                    game_players.append(player)
                
                game_states.append((state, None))  # Will fill in the move later
                
                # Choose move
                valid_moves = game.get_valid_moves()
                if not valid_moves:  # Check if valid_moves is empty
                    break
                    
                move = ai.get_move(game, valid_moves)
                
                # Update the target for the previous state (for opponent prediction)
                if len(game_states) > 1:
                    row, col = move
                    target = np.zeros(board_size * board_size)
                    target[row * board_size + col] = 1
                    game_states[-2] = (game_states[-2][0], target)
                
                # Make move
                row, col = move
                game.make_move(row, col)
            
            # After game ends, assign rewards to patterns
            if game.winner != 0:  # If not a draw
                for i, (pattern, player) in enumerate(zip(game_patterns, game_players)):
                    if player == game.winner:
                        reward = 1.0  # Reward for winner
                    else:
                        reward = -1.0  # Penalty for loser
                    
                    # Remember pattern and reward
                    ai.remember(pattern, reward)
                
                # Remember move sequence - tylko raz, po pętli for
                ai.remember_move_sequence(game.move_history)
            
            # Train opponent prediction model with the collected states and moves
            for state, target in game_states:
                if target is not None:  # Skip the last state which has no next move
                    ai.opponent_memory.append((state, target))
            
            # Every few games, train on collected experiences
            if (game_num + 1) % 5 == 0 or game_num == num_games - 1:
                if len(ai.memory) >= 128:
                    ai.replay(batch_size=min(512, len(ai.memory)), epochs=3)
        
        print("Self-play training completed!")
        return ai
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Return the AI object even if there was an error
        return ai




def train_model_with_rules_and_gameplay(board_size=15, initial_epochs=30, num_patterns=10000, 
                                        num_expert_patterns=5000, self_play_games=50):
    """
    Comprehensive function to train the model:
    1. First trains based on strategic rules
    2. Then refines knowledge through self-play
    """
    print("=== STARTING COMPREHENSIVE MODEL TRAINING ===")
    
            
     # Step 1: Initialize the AI model
    print("\n--- Initializing AI model ---")
    ai = GomokuAI(board_size)
    
    # Step 2: Training based on strategic rules
    print(f"\n--- Training based on strategic rules ({initial_epochs} epochs, {num_patterns} patterns, {num_expert_patterns} expert patterns) ---")
    
    # Train on strategic and expert patterns
    success = train_initial_model(ai, epochs=initial_epochs, num_patterns=num_patterns, num_expert_patterns=num_expert_patterns)
    
    if not success:
        print("Error during rule-based training. Trying with fewer patterns...")
        success = train_initial_model(ai, epochs=initial_epochs, num_patterns=num_patterns//2, num_expert_patterns=num_expert_patterns//2)
        if not success:
            print("Rule-based training failed. Proceeding with untrained model.")
    else:
        print("Completed training based on strategic rules")
    
    # Save model after rule-based training
    if not os.path.exists('models'):
        os.makedirs('models')
    timestamp = int(time.time())
    rules_model_path = f'models/gomoku_rules_trained_{board_size}x{board_size}_{timestamp}.h5'
    ai.save_model(rules_model_path)
    print(f"Model after rule-based training saved as: {rules_model_path}")
    
    # Step 3: Training through self-play
    print(f"\n--- Training through self-play ({self_play_games} games) ---")
    
    # Track game statistics
    game_stats = {'wins_player1': 0, 'wins_player2': 0, 'draws': 0}
    
    # Self-play training in batches
    batch_size = 10
    for batch in range(0, self_play_games, batch_size):
        batch_games = min(batch_size, self_play_games - batch)
        print(f"\nStarting batch of {batch_games} self-play games ({batch+1}-{batch+batch_games} of {self_play_games})...")
        
        try:
            # Train through self-play
            ai = train_by_self_play(ai, num_games=batch_games, board_size=board_size)
       
            # Save intermediate model after each batch
            intermediate_model_path = f'models/gomoku_intermediate_{board_size}x{board_size}_{timestamp}_batch{batch//batch_size+1}.h5'
            ai.save_model(intermediate_model_path)
            print(f"Intermediate model saved as: {intermediate_model_path}")
            
            # Update game statistics by playing evaluation games
            print("Playing evaluation games to measure progress...")
            eval_stats = evaluate_ai_strength(ai, num_games=5, board_size=board_size)
            game_stats['wins_player1'] += eval_stats['wins_player1']
            game_stats['wins_player2'] += eval_stats['wins_player2']
            game_stats['draws'] += eval_stats['draws']
            
            print(f"Current statistics - X wins: {game_stats['wins_player1']}, O wins: {game_stats['wins_player2']}, Draws: {game_stats['draws']}")
            print(f"Current epsilon (randomness): {ai.epsilon:.4f}")
        except Exception as e:
                    print(f"Error in batch training: {str(e)}")
                    print("Continuing with next batch...")
                    continue

     
    
    # Step 4: Save final model
    final_model_path = f'models/gomoku_fully_trained_{board_size}x{board_size}_{timestamp}.h5'
    ai.save_model(final_model_path)
    print(f"Final model saved as: {final_model_path}")
    
    print("\n=== COMPREHENSIVE MODEL TRAINING COMPLETED ===")
    print(f"Final statistics:")
    print(f"X wins: {game_stats['wins_player1']}, O wins: {game_stats['wins_player2']}, Draws: {game_stats['draws']}")
    
    return ai, rules_model_path, final_model_path


def evaluate_ai_strength(ai, num_games=10, board_size=15):
    """Evaluates AI strength by playing games against itself with different exploration rates."""
    game = GomokuGame(board_size)
    stats = {'wins_player1': 0, 'wins_player2': 0, 'draws': 0}
    
    # Store original epsilon
    original_epsilon = ai.epsilon
    
    # Set very low epsilon for evaluation
    ai.epsilon = 0.05
    
    for game_num in range(num_games):
        game.reset()
        
        # First move near the center with some randomness
        mid = board_size // 2
        offset_r = random.randint(-2, 2)
        offset_c = random.randint(-2, 2)
        game.make_move(mid + offset_r, mid + offset_c)
        
        move_count = 0
        max_moves = board_size * board_size
        
        # Play the game
        while not game.game_over and move_count < max_moves:
            move_count += 1
            
            # Get best move
            move = ai.get_best_move(game)
            
            # Make move
            row, col = move
            game.make_move(row, col)
        
        # Record result
        if game.winner == 0:
            stats['draws'] += 1
        elif game.winner == 1:
            stats['wins_player1'] += 1
        else:
            stats['wins_player2'] += 1
    
    # Restore original epsilon
    ai.epsilon = original_epsilon
    
    return stats


def play_sample_game(ai, board_size=15):
    """Plays a sample game with the trained network."""
    game = GomokuGame(board_size)
    
    print("Starting sample game...")
    game.print_board()
    
    # First move near the center with some randomness
    mid = board_size // 2
    offset_r = random.randint(-2, 2)
    offset_c = random.randint(-2, 2)
    game.make_move(mid + offset_r, mid + offset_c)
    print(f"First move: {mid + offset_r}, {mid + offset_c}")
    game.print_board()
    
    # Store original epsilon and set to 0 for deterministic play
    original_epsilon = ai.epsilon
    ai.epsilon = 0.0
    
    move_count = 1
    while not game.game_over:
        # Get move from network
        move = ai.get_best_move(game)
        
        # Make move
        row, col = move
        game.make_move(row, col)
        
        move_count += 1
        print(f"Move {move_count}: Player {3 - game.current_player} plays: {row}, {col}")
        game.print_board()
    
    # Restore original epsilon
    ai.epsilon = original_epsilon
    
    # Display result
    if game.winner == 0:
        print("Game ended in a draw!")
    else:
        print(f"Player {game.winner} wins!")


def play_against_ai(ai, board_size=15):
    """Allows a player to play against the AI."""
    game = GomokuGame(board_size)
    human_player = int(input("Choose your player (1 for X, 2 for O): "))
    ai_player = 3 - human_player
    
    # Reset game
    game.reset()
    game.print_board()
    
    # Temporarily reduce epsilon for game against human
    original_epsilon = ai.epsilon
    ai.epsilon = 0.0  # Turn off randomness for game against human
    
    # If AI goes first, make a move near center
    if game.current_player == ai_player:
        mid = board_size // 2
        offset_r = random.randint(-2, 2)
        offset_c = random.randint(-2, 2)
        game.make_move(mid + offset_r, mid + offset_c)
        print(f"AI plays: {mid + offset_r}, {mid + offset_c}")
        game.print_board()
    
    # Main game loop
    while not game.game_over:
        if game.current_player == human_player:
            # Human move
            valid_move = False
            while not valid_move:
                try:
                    row = int(input("Enter row: "))
                    col = int(input("Enter column: "))
                    valid_move = game.make_move(row, col)
                    if not valid_move:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter valid numbers.")
        else:
            # AI move
            print("AI is thinking...")
            move = ai.get_best_move(game)
            row, col = move
            game.make_move(row, col)
            print(f"AI plays: {row}, {col}")
        
        # Display board after move
        game.print_board()
    
    # Display result
    if game.winner == 0:
        print("Game ended in a draw!")
    elif game.winner == human_player:
        print("Congratulations! You won!")
    else:
        print("AI won the game!")
    
    # Restore original epsilon
    ai.epsilon = original_epsilon



def export_game_for_gui(game, filepath):
    """Exports a game's move history to a JSON file for GUI replay."""
    game_data = {
        "board_size": game.board_size,
        "moves": [(move[0], move[1], move[2]) for move in game.move_history],
        "winner": game.winner
    }
    
    with open(filepath, 'w') as f:
        json.dump(game_data, f)
    
    print(f"Game exported to {filepath}")


def main():
    ai = None
    board_size = 15  # Fixed board size 15x15
    
    while True:
        print("\nGomoku AI - Menu:")
        print("1. Train new model")
        print("2. Train model by self-play")
        print("3. Complete training (rules + self-play)")
        print("4. Save model")
        print("5. Load model")
        print("6. Play sample game")
        print("7. Play against AI")
        print("8. Export model for GUI")
        print("9. Exit")
        
        choice = input("Enter your choice (1-9): ")
        
        if choice == '1':
            try:
                epochs = int(input("Enter number of training epochs (recommended 20-50): "))
                patterns = int(input("Enter number of training patterns (recommended 10000-50000): "))
                expert_patterns = int(input("Enter number of expert patterns (recommended 5000-10000): "))
                
                ai = GomokuAI(board_size)
                success = train_initial_model(ai, epochs=epochs, num_patterns=patterns, num_expert_patterns=expert_patterns)
                
                if not success:
                    print("Training failed. Please try again.")
            except ValueError:
                print("Please enter valid numbers.")
            except Exception as e:
                print(f"Error during training: {e}")
                
        elif choice == '2':
            if ai is None:
                try:
                    print("No model available. Training a basic model first...")
                    ai = GomokuAI(board_size)
                    train_initial_model(ai, epochs=10, num_patterns=5000, num_expert_patterns=2000)
                except Exception as e:
                    print(f"Error during initial training: {e}")
                    continue
            
            try:
                num_games = int(input("Enter number of self-play games (recommended 20-100): "))
                train_by_self_play(ai, num_games=num_games, board_size=board_size)
            except ValueError:
                print("Please enter a valid number.")
            except Exception as e:
                print(f"Error during self-play training: {e}")
        
        elif choice == '3':  # Comprehensive training
            try:
                initial_epochs = int(input("Enter number of rule-based training epochs (recommended 20-50): "))
                num_patterns = int(input("Enter number of training patterns (recommended 10000-50000): "))
                expert_patterns = int(input("Enter number of expert patterns (recommended 5000-10000): "))
                self_play_games = int(input("Enter number of self-play games (recommended 50-200): "))
                             
                ai, rules_model_path, final_model_path = train_model_with_rules_and_gameplay(
                    board_size=board_size,
                    initial_epochs=initial_epochs,
                    num_patterns=num_patterns,
                    num_expert_patterns=expert_patterns,
                    self_play_games=self_play_games
                )
                
                print(f"Training completed!")
                print(f"Rules-trained model: {rules_model_path}")
                print(f"Final model: {final_model_path}")
                
            except ValueError:
                print("Please enter valid numbers.")
            except Exception as e:
                print(f"Error during training: {e}")
                
        elif choice == '4':  # Save model
            if ai is None:
                print("No model to save. Please train a model first.")
            else:
                try:
                    if not os.path.exists('models'):
                        os.makedirs('models')
                    timestamp = int(time.time())
                    filepath = f'models/gomoku_ai_{board_size}x{board_size}_{timestamp}.h5'
                    ai.save_model(filepath)
                except Exception as e:
                    print(f"Error saving model: {e}")
                    
        elif choice == '5':  # Load model
            try:
                filepath = input("Enter model filepath: ")
                if not os.path.exists(filepath):
                    print(f"File not found: {filepath}")
                else:
                    if ai is None:
                        ai = GomokuAI(board_size)
                    success = ai.load_model(filepath)
                    if not success:
                        print("Failed to load model. Please check the file format.")
            except Exception as e:
                print(f"Error loading model: {e}")
                
        elif choice == '6':  # Sample game
            if ai is None:
                print("No model available. Please train or load a model first.")
            else:
                try:
                    play_sample_game(ai, board_size)
                except Exception as e:
                    print(f"Error during sample game: {e}")
                    
        elif choice == '7':  # Play against AI
            if ai is None:
                print("No model available. Please train or load a model first.")
            else:
                try:
                    play_against_ai(ai, board_size)
                except Exception as e:
                    print(f"Error during game: {e}")
        
        elif choice == '8':  # Export model for GUI
            if ai is None:
                print("No model available. Please train or load a model first.")
            else:
                try:
                    if not os.path.exists('exports'):
                        os.makedirs('exports')
                    timestamp = int(time.time())
                    filepath = f'exports/gomoku_ai_{board_size}x{board_size}_{timestamp}.h5'
                    ai.save_model(filepath)
                    print(f"Model exported to {filepath}")
                    
                    # Create a sample game for GUI visualization
                    print("Creating sample game for GUI visualization...")
                    game = GomokuGame(board_size)
                    game.reset()
                    
                    # First move near the center
                    mid = board_size // 2
                    game.make_move(mid, mid)
                    
                    # Play a sample game
                    move_count = 1
                    while not game.game_over and move_count < 30:  # Limit to 30 moves for demo
                        move = ai.get_best_move(game)
                        row, col = move
                        game.make_move(row, col)
                        move_count += 1
                    
                    # Export game
                    game_filepath = f'exports/sample_game_{board_size}x{board_size}_{timestamp}.json'
                    export_game_for_gui(game, game_filepath)
                    
                except Exception as e:
                    print(f"Error exporting model: {e}")
                    
        elif choice == '9':  # Exit
            print("Exiting program. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
