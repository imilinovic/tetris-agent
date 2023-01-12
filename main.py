from typing import List, Tuple

import logging
import random
import time

import numpy as np


from tetris_app import TetrisApp, rotate_clockwise, check_collision
from tetris_app_no_gui import TetrisAppNoGUI

class TetrisAgent():
    def __init__(self, tetrisApp: TetrisApp, ):
        np.set_printoptions(suppress=True)
        self.tetrisApp = tetrisApp
        self.generation_id = 1
        self.generation_size = 50
        self.best_fitness = 0
        self.current_id = 0
        self.current_run = 0
        self.current_run_score_sum = 0
        self.runs_per_chromosome = 3
        self.weights = (np.random.rand(self.generation_size, 8) - 0.5)
        #self.weights[0] = [1.38049889,-3.1927763,9.43484846,0.00224655,-2.98497962]
        self.fitness = []
        self.mutation_coefficient = 0.01


    def get_height_difference(self, board: np.ndarray, debug=False) -> int:
        """ Returns sum of absolute height differences between each neighbor column """
        heights = (board != 0).argmax(axis=0)
        return np.sum(np.abs(np.diff(heights)))


    def get_total_height(self, board: np.ndarray, debug=False) -> int:
        """ Returns total sum of column heights """
        heights = (board != 0).argmax(axis=0)
        return np.sum(board.shape[0] - 1 - heights)


    def get_number_of_holes(self, board: np.ndarray, debug=False) -> int:
        """ Hole is defined as empty cell lower than column height """
        heights = (board != 0).argmax(axis=0)
        empty = (board == 0)
        if debug:
            logging.info(heights)
            logging.info(empty)
        sum = 0
        for column in range(board.shape[1]):
            sum += np.sum(empty[heights[column]:, column], axis=0)
        return sum


    def get_square_of_number_of_holes(self, board: np.ndarray, debug=False) -> int:
        """ Returns square of number of holes """
        holes = self.get_number_of_holes(board, debug)
        return np.square(holes)


    def get_number_of_columns_with_holes(self, board: np.ndarray, debug=False) -> int:
        """
        Returns number of columns with at least one hole
        Hole is defined as empty cell lower than column height
        """
        heights = (board != 0).argmax(axis=0)
        empty = (board == 0)
        if debug:
            logging.info(heights)
            logging.info(empty)
        sum = 0
        for column in range(board.shape[1]):
            if np.sum(empty[heights[column]:, column], axis=0) > 0:
                sum += 1
        return sum


    def get_max_height_difference(self, board: np.ndarray, debug=False) -> int:
        """ Returns max absolute height difference between columns """
        heights = (board != 0).argmax(axis=0)
        if debug:
            print(heights)
        return np.amax(heights) - np.amin(heights)


    def get_mul_hole_height_diff(self, board: np.ndarray, debug=False) -> int:
        """ Returns product of number of holes and sum of absolute height differences between each neighbor column """
        holes = self.get_number_of_holes(board, debug)
        height = self.get_height_difference(board, debug)
        return np.multiply(holes, height)


    def get_board_parameters(self, board, debug=False) -> np.ndarray:
        return np.array([
            self.get_height_difference(board, debug),
            self.get_total_height(board, debug),
            self.get_number_of_holes(board, debug),
            self.get_square_of_number_of_holes(board, debug),
            self.get_number_of_columns_with_holes(board, debug),
            self.get_max_height_difference(board, debug),
            self.get_mul_hole_height_diff(board, debug),
        ])


    def simulate_move(self, column: int, rotation_cnt: int) -> np.ndarray:
        stone = self.state['stone']
        if check_collision(self.state['board'], stone, (self.state['stone_x'], self.state['stone_y'])):
            return None # unable to rotate stone

        for i in range(rotation_cnt):
            stone = rotate_clockwise(stone)
            if check_collision(self.state['board'], stone, (self.state['stone_x'], self.state['stone_y'])):
               return None # unable to rotate stone

        if check_collision(self.state['board'], stone, (column, 0)):
            return None # unable to go there with rotated stone

        stone = np.array(stone)
        board = np.copy(self.state['board'])
        heights = (board != 0).argmax(axis=0)
        min_height = np.min(heights[column:column+stone.shape[1]])

        height_offset = stone.shape[0]
        start_height = max(min_height - height_offset, 0)

        while not check_collision(board, stone, (column, start_height+1)):
            start_height += 1

        for i in range(stone.shape[0]):
            for j in range(stone.shape[1]):
                if stone[i][j]:
                    board[start_height + i][column + j] = stone[i][j]

        return board


    def clean_board(self, board: np.ndarray) -> int:
        """Modifies the board by removing full rows and returns number of cleared rows"""
        non_zero_rows = ((board != 0).sum(1) != board.shape[1])
        non_zero_rows[-1] = True
        new_board = board[non_zero_rows]
        
        rows_cleared = board.shape[0] - new_board.shape[0]
        empty_rows = np.zeros((rows_cleared, board.shape[1]), dtype=int)

        if rows_cleared > 0:
            board = np.concatenate((empty_rows, new_board), axis=0)
        return rows_cleared


    def evaluate_move(self, column: int, rotation_cnt: int) -> float:
        if (board := self.simulate_move(column, rotation_cnt)) is None:
            return 1e9

        rows_cleared = self.clean_board(board)
        input_layer = self.get_board_parameters(board)
        input_layer = np.append(input_layer, [rows_cleared*100])
        
        return np.matmul(input_layer, self.weights[self.current_id])

    def find_optimal_move(self) -> Tuple[int, int]:
        min_value = 1e9
        move = (-1, 0)
        for column in range(self.state['board'].shape[1]):
            for rotation_cnt in range(4):
                value = self.evaluate_move(column, rotation_cnt)
                if value < min_value:
                    min_value = value
                    move = (column, rotation_cnt)

        return move
    

    def play_move(self, move: Tuple[int, int]):
        moves = []
        for _ in range(move[1]):
            moves.append('UP')
        stone_x = self.state['stone_x']
        if stone_x - move[0] > 0:
            for _ in range(stone_x - move[0]):
                moves.append('LEFT')
        else:
            for _ in range(move[0] - stone_x):
                moves.append('RIGHT')
        moves.append('DOWN')
        self.tetrisApp.add_actions(moves)


    def mutate(self, id):
        how_many = random.randint(1, 5)
        which = random.sample(range(0, 5), how_many)

        for idx in which:
            self.weights[id][idx] += np.random.normal() * (id*self.mutation_coefficient)


    def crossover(self, i):
        parent_choice = []
        for i in range(self.generation_size):
            parent_choice += [i] * (self.generation_size-i)         

        parent1 = random.choice(parent_choice)
        while (parent2 := random.choice(parent_choice)) == parent1:
            pass
        
        for j in range(len(self.weights[i])):
            if random.random() < 0.5:
                self.weights[i][j] = self.weights[parent1][j]
            else:
                self.weights[i][j] = self.weights[parent2][j]

    def create_next_generation(self):
        self.fitness = sorted(self.fitness, reverse=True)
        logging.info(f"\n\n\nGeneration: {self.generation_id}")
        logging.info(f"Best fitness: {self.best_fitness}")
        weights = self.weights
        for i in range(len(self.weights)):
            weights[i] = self.weights[self.fitness[i][1]]
        self.weights = weights
        logging.info(self.weights)
        logging.info(self.fitness)
        
        keep_id = np.ceil(0.2*self.generation_size).astype(int)
        for i in range(keep_id+1, self.generation_size):
            self.crossover(i)

        #logging.info(f"Weights after parenting: {self.weights}")
        for i in range(keep_id+1, self.generation_size):
            if random.random()+0.1 < i/self.generation_size:
                self.mutate(i)

        #logging.info(f"Weights after mutating: {self.weights}")

        self.generation_id += 1
        self.current_id = keep_id+1
        self.fitness = self.fitness[:keep_id+1]
        #print(self.fitness, "FITENS", keep_id)


    def start(self):
        self.tetrisApp.init()

        while(1):
            
            self.state = self.tetrisApp.get_state()
            if self.state["gameover"] and not self.tetrisApp.actions:
                self.current_run += 1
                self.current_run_score_sum += self.state["score"]

                if self.current_run == self.runs_per_chromosome:
                    self.fitness.append((self.current_run_score_sum/self.runs_per_chromosome, self.current_id))
                    if self.current_run_score_sum/self.runs_per_chromosome > self.best_fitness:
                        self.best_fitness = self.current_run_score_sum/self.runs_per_chromosome
                    self.current_id += 1
                    self.current_run = 0
                    self.current_run_score_sum = 0
                    logging.info(self.fitness[-1])

                if self.current_id == self.generation_size:
                    self.create_next_generation()
                self.tetrisApp.add_actions(["SPACE"])
            elif not self.state["gameover"] and not self.tetrisApp.actions:
                self.tetrisApp.tick()
                optimal_move = self.find_optimal_move()
                self.play_move(optimal_move)
            
            self.tetrisApp.tick()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app = TetrisAppNoGUI()
    agent = TetrisAgent(app)
    agent.start()
