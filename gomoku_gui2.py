import os
import sys
import random
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox
from tensorflow.keras.models import load_model
import tensorflow as tf

# Wyłączenie ostrzeżeń TensorFlow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class GomokuGame:
    def __init__(self, board_size=15):  # Zmiana rozmiaru planszy na 15x15
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1  # 1 for 'X', 2 for 'O'
        self.game_over = False
        self.winner = None
        
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        
    def make_move(self, row, col):
        if self.game_over or row < 0 or row >= self.board_size or col < 0 or col >= self.board_size or self.board[row, col] != 0:
            return False
        
        self.board[row, col] = self.current_player
        
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
    
    def get_valid_moves(self):
        return [(i, j) for i in range(self.board_size) for j in range(self.board_size) if self.board[i, j] == 0]


class GomokuAI:
    def __init__(self, model_path, board_size=15):  # Zmiana rozmiaru planszy na 15x15
        self.board_size = board_size
        self.model = load_model(model_path)
        
    def extract_patterns(self, board, player):
        """Ekstrahuje wszystkie możliwe wzorce 7x7 z planszy."""
        opponent = 3 - player
        patterns = []
        positions = []
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == 0:  # tylko puste pola
                    # Wytnij fragment 7x7 z centrum w (i,j)
                    pattern = np.zeros((7, 7, 3))
                    for r in range(-3, 4):
                        for c in range(-3, 4):
                            if 0 <= i+r < self.board_size and 0 <= j+c < self.board_size:
                                if board[i+r, j+c] == player:
                                    pattern[r+3, c+3, 0] = 1  # kanał gracza
                                elif board[i+r, j+c] == opponent:
                                    pattern[r+3, c+3, 1] = 1  # kanał przeciwnika
                    
                    # Ustaw wskaźnik aktualnego gracza w trzecim kanale
                    pattern[:, :, 2] = player - 1
                    
                    patterns.append(pattern)
                    positions.append((i, j))
        
        return np.array(patterns), positions
    
    def get_best_move(self, game):
        """Zwraca najlepszy ruch na podstawie oceny wzorców i reguł strategicznych."""
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        
        player = game.current_player
        opponent = 3 - player
        board = game.board.copy()
        
        # 1. Sprawdź ruchy wygrywające
        for move in valid_moves:
            row, col = move
            board_copy = board.copy()
            board_copy[row, col] = player
            temp_game = GomokuGame(self.board_size)
            temp_game.board = board_copy
            temp_game.current_player = player
            if temp_game.check_win(row, col):
                print(f"Found winning move: {row}, {col}")
                return move
        
        # 2. Sprawdź ruchy blokujące wygraną przeciwnika
        for move in valid_moves:
            row, col = move
            board_copy = board.copy()
            board_copy[row, col] = opponent
            temp_game = GomokuGame(self.board_size)
            temp_game.board = board_copy
            temp_game.current_player = opponent
            if temp_game.check_win(row, col):
                print(f"Found blocking move: {row}, {col}")
                return move
        
        # 3. Sprawdź ruchy tworzące otwarte czwórki
        for move in valid_moves:
            row, col = move
            if self._creates_open_four(board, row, col, player):
                print(f"Found move creating open four: {row}, {col}")
                return move
        
        # 4. Sprawdź ruchy blokujące otwarte czwórki przeciwnika
        for move in valid_moves:
            row, col = move
            if self._creates_open_four(board, row, col, opponent):
                print(f"Found move blocking open four: {row}, {col}")
                return move
        
        # 5. Użyj sieci neuronowej do oceny pozostałych ruchów
        patterns, positions = self.extract_patterns(board, player)
        
        if len(patterns) == 0:
            return valid_moves[0]
        
        # Filtruj tylko dozwolone ruchy
        valid_indices = []
        for i, pos in enumerate(positions):
            if pos in valid_moves:
                valid_indices.append(i)
        
        if not valid_indices:
            return valid_moves[0]
        
        # Oceń wszystkie wzorce
        evaluations = self.model.predict(patterns, verbose=0).flatten()
        
        # Dodaj preferencję dla ruchów bliżej środka planszy
        center = self.board_size // 2
        for i, pos in enumerate(positions):
            r, c = pos
            # Odległość od środka (0 w środku, większa na brzegach)
            distance = abs(r - center) + abs(c - center)
            # Zmniejsz ocenę dla ruchów dalej od środka
            evaluations[i] -= distance * 0.01
        
        # Wyświetl top 5 ruchów z ich ocenami
        top_indices = np.argsort(evaluations)[-5:]
        print("Top 5 moves:")
        for idx in reversed(top_indices):
            print(f"Position {positions[idx]}: score {evaluations[idx]}")
        
        # Wybierz losowo jeden z 5 najlepszych ruchów
        top_n = min(5, len(valid_indices))
        
        # Oblicz oceny dla dozwolonych ruchów
        valid_evaluations = [evaluations[i] for i in valid_indices]
        # Znajdź indeksy najlepszych ocen
        best_indices = np.argsort(valid_evaluations)[-top_n:]
        # Wybierz losowo jeden z najlepszych indeksów
        chosen_valid_idx = np.random.choice(best_indices)
        # Przekonwertuj na oryginalny indeks
        chosen_idx = valid_indices[chosen_valid_idx]
        
        return positions[chosen_idx]

    def _creates_open_four(self, board, row, col, player):
        """Sprawdza czy ruch tworzy otwartą czwórkę (4 w linii z pustym polem na obu końcach)."""
        board_copy = board.copy()
        board_copy[row, col] = player
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # poziomo, pionowo, ukośnie
        
        for dr, dc in directions:
            count = 1  # Licznik pionków w linii
            open_ends = 0  # Licznik otwartych końców
            
            # Sprawdź w jedną stronę
            r, c = row + dr, col + dc
            while 0 <= r < len(board_copy) and 0 <= c < len(board_copy) and board_copy[r, c] == player:
                count += 1
                r += dr
                c += dc
            
            # Sprawdź czy koniec jest otwarty
            if 0 <= r < len(board_copy) and 0 <= c < len(board_copy) and board_copy[r, c] == 0:
                open_ends += 1
            
            # Sprawdź w drugą stronę
            r, c = row - dr, col - dc
            while 0 <= r < len(board_copy) and 0 <= c < len(board_copy) and board_copy[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            
            # Sprawdź czy drugi koniec jest otwarty
            if 0 <= r < len(board_copy) and 0 <= c < len(board_copy) and board_copy[r, c] == 0:
                open_ends += 1
            
            # Jeśli mamy 4 w linii z co najmniej jednym otwartym końcem
            if count == 4 and open_ends >= 1:
                return True
        
        return False


class GomokuGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gomoku Game")
        self.root.geometry("900x800")  # Zwiększenie rozmiaru okna dla planszy 15x15
        self.root.resizable(False, False)
        
        # Inicjalizacja zmiennych
        self.board_size = 15  # Zmiana rozmiaru planszy na 15x15
        self.cell_size = 40  # Zmniejszenie rozmiaru komórki dla większej planszy
        self.game = GomokuGame(self.board_size)
        self.ai = None  # Będzie zainicjalizowany po załadowaniu modelu
        self.human_player = 1  # 1 for 'X', 2 for 'O'
        self.ai_player = 2
        
        # Kolory
        self.board_color = "#EEEEEE"  # Domyślny kolor planszy (jasny szary)
        self.x_color = "#000000"  # Domyślny kolor X (czarny)
        self.o_color = "#FF0000"  # Domyślny kolor O (czerwony)
        
        # Tworzenie interfejsu
        self.create_widgets()
        
        # Inicjalizacja gry
        self.reset_game()
        
    def create_widgets(self):
        # Główny kontener
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel kontrolny
        control_frame = ttk.Frame(main_frame, padding=5)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Przyciski
        ttk.Button(control_frame, text="New Game", command=self.new_game).pack(side=tk.LEFT, padx=5)
        
        # Przycisk dodawania modelu
        ttk.Button(control_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        
        # Wybór koloru planszy
        ttk.Button(control_frame, text="Board Color", command=self.change_board_color).pack(side=tk.LEFT, padx=5)
        
        # Wybór gracza (X lub O)
        player_frame = ttk.Frame(control_frame)
        player_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(player_frame, text="Play as:").pack(side=tk.LEFT)
        self.player_var = tk.StringVar(value="X")
        player_menu = ttk.OptionMenu(player_frame, self.player_var, "X", "X", "O", command=self.change_player)
        player_menu.pack(side=tk.LEFT)
        
        # Przyciski zmiany kolorów X i O
        ttk.Button(control_frame, text="X Color", command=lambda: self.change_piece_color('X')).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="O Color", command=lambda: self.change_piece_color('O')).pack(side=tk.LEFT, padx=5)
        
        # Przycisk wyjścia
        ttk.Button(control_frame, text="Exit", command=self.root.destroy).pack(side=tk.RIGHT, padx=5)
        
                # Status gry
        self.status_var = tk.StringVar(value="Please load an AI model to start.")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, font=('Arial', 12))
        status_label.pack(side=tk.TOP, pady=5)
        
        # Plansza do gry
        board_frame = ttk.Frame(main_frame, padding=5)
        board_frame.pack(side=tk.TOP, pady=10)
        
        # Canvas do rysowania planszy
        canvas_size = self.board_size * self.cell_size + 1
        self.canvas = tk.Canvas(board_frame, width=canvas_size, height=canvas_size, bg=self.board_color)
        self.canvas.pack()
        
        # Rysowanie linii planszy
        self.draw_board()
        
        # Obsługa kliknięć
        self.canvas.bind("<Button-1>", self.handle_click)
        
    def draw_board(self):
        # Czyszczenie planszy
        self.canvas.delete("all")
        
        # Rysowanie linii
        for i in range(self.board_size):
            # Linie poziome
            self.canvas.create_line(
                0, i * self.cell_size, 
                self.board_size * self.cell_size, i * self.cell_size,
                fill="black"
            )
            # Linie pionowe
            self.canvas.create_line(
                i * self.cell_size, 0, 
                i * self.cell_size, self.board_size * self.cell_size,
                fill="black"
            )
        
        # Rysowanie kamieni
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.game.board[i, j] == 1:  # X
                    self.draw_x(i, j)
                elif self.game.board[i, j] == 2:  # O
                    self.draw_o(i, j)
        
        # Oznaczenie punktów Hoshi (punkty orientacyjne na planszy)
        hoshi_points = []
        if self.board_size == 15:
            hoshi_points = [(3, 3), (3, 7), (3, 11), (7, 3), (7, 7), (7, 11), (11, 3), (11, 7), (11, 11)]
        
        for point in hoshi_points:
            i, j = point
            x = j * self.cell_size
            y = i * self.cell_size
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")
    
    def draw_x(self, row, col):
        # Oblicz środek komórki
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        
        # Rysuj X jako tekst wyśrodkowany w komórce
        self.canvas.create_text(
            x, y, 
            text="X", 
            fill=self.x_color, 
            font=("Arial", int(self.cell_size * 0.6)),
            anchor="center"  # Ważne - ustawia punkt zakotwiczenia na środek tekstu
    )
    
    def draw_o(self, row, col):
        # Oblicz środek komórki
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        
        # Rysuj O jako okrąg wyśrodkowany w komórce
        radius = self.cell_size // 2 - 4  # Nieco mniejszy promień, aby nie dotykał linii siatki
        self.canvas.create_oval(
            x - radius, y - radius, 
            x + radius, y + radius, 
            outline=self.o_color, 
            width=2
        )
    
    def handle_click(self, event):
        if self.ai is None:
            messagebox.showinfo("AI Not Loaded", "Please load an AI model first.")
            return
            
        if self.game.game_over:
            return
            
        if self.game.current_player != self.human_player:
            return
            
        # Konwersja współrzędnych kliknięcia na indeksy planszy
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            # Wykonanie ruchu gracza
            if self.game.make_move(row, col):
                self.draw_board()
                self.update_status()
                
                # Sprawdzenie czy gra się zakończyła
                if self.game.game_over:
                    return
                
                # Ruch AI
                self.root.after(500, self.ai_move)
    
    def ai_move(self):
        if self.game.game_over or self.game.current_player != self.ai_player:
            return
            
        # Pobranie ruchu od AI
        move = self.ai.get_best_move(self.game)
        
        if move:
            row, col = move
            self.game.make_move(row, col)
            self.draw_board()
            self.update_status()
    
    def update_status(self):
        if self.game.game_over:
            if self.game.winner == 0:
                self.status_var.set("Game ended in a draw!")
            elif self.game.winner == self.human_player:
                self.status_var.set("You won!")
            else:
                self.status_var.set("AI won!")
        else:
            if self.game.current_player == self.human_player:
                self.status_var.set("Your turn")
            else:
                self.status_var.set("AI is thinking...")
    
    def new_game(self):
        if self.ai is None:
            messagebox.showinfo("AI Not Loaded", "Please load an AI model first.")
            return
            
        self.reset_game()
        
        # Jeśli AI zaczyna, wykonaj pierwszy ruch
        if self.game.current_player == self.ai_player:
            self.ai_move()
    
    def reset_game(self):
        self.game.reset()
        self.draw_board()
        self.update_status()
    
    def load_model(self):
        try:
            model_path = filedialog.askopenfilename(
                title="Select AI Model",
                filetypes=[("H5 Files", "*.h5"), ("All Files", "*.*")]
            )
            
            if model_path:
                self.ai = GomokuAI(model_path, self.board_size)
                self.status_var.set("Model loaded successfully. Start a new game.")
                messagebox.showinfo("Success", "AI model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def change_board_color(self):
        color = colorchooser.askcolor(initialcolor=self.board_color)
        if color[1]:
            self.board_color = color[1]
            self.canvas.config(bg=self.board_color)
            self.draw_board()
    
    def change_piece_color(self, piece):
        if piece == 'X':
            color = colorchooser.askcolor(initialcolor=self.x_color)
            if color[1]:
                self.x_color = color[1]
        else:  # 'O'
            color = colorchooser.askcolor(initialcolor=self.o_color)
            if color[1]:
                self.o_color = color[1]
        
        self.draw_board()
    
    def change_player(self, selection):
        if selection == "X":
            self.human_player = 1
            self.ai_player = 2
        else:  # "O"
            self.human_player = 2
            self.ai_player = 1
        
        # Resetuj grę po zmianie gracza
        self.reset_game()
        
        # Jeśli AI zaczyna, wykonaj pierwszy ruch
        if self.game.current_player == self.ai_player:
            self.ai_move()


def main():
    root = tk.Tk()
    app = GomokuGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

