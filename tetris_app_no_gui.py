'''
This script is used apply genes to from the training script
'''

from random import randrange as rand
import numpy
# The configuration
config = {
    'cell_size':  20,
    'cols':    10,
    'rows':    24,
    'delay':  750,
    'maxfps':  30
}

colors = [
    (0,   0,   0  ),
    (255, 0,   0  ),
    (0,   150, 0  ),
    (0,   0,   255),
    (255, 120, 0  ),
    (255, 255, 0  ),
    (180, 0,   255),
    (0,   220, 220)
]

# Define the shapes of the single parts
tetris_shapes = [
    [[1, 1, 1],
    [0, 1, 0]],

    [[0, 2, 2],
    [2, 2, 0]],

    [[3, 3, 0],
    [0, 3, 3]],

    [[4, 0, 0],
    [4, 4, 4]],

    [[0, 0, 5],
    [5, 5, 5]],

    [[6, 6, 6, 6]],

    [[7, 7],
    [7, 7]]
]

    # Rotates a shape clockwise
def rotate_clockwise(shape):
    return [ [ shape[y][x]
        for y in range(len(shape)) ]
        for x in range(len(shape[0]) - 1, -1, -1) ]

# checks if there is a collision in any direction
def check_collision(board, shape, offset):
    off_x, off_y = offset
    for cy, row in enumerate(shape):
        for cx, cell in enumerate(row):
            try:
                if cell and board[ cy + off_y ][ cx + off_x ]:
                    return True
            except IndexError:
                    return True
    return False

# clearing a row for getting points
def remove_row(board, row):
    del board[row]
    return [[0 for i in range(config['cols'])]] + board

# Used for adding a stone to the board
def join_matrixes(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            mat1[cy+off_y-1  ][cx+off_x] += val
    return mat1

# new game
def new_board():
    board = [ [ 0 for x in range(config['cols']) ]
        for y in range(config['rows']) ]
    board += [[ 1 for x in range(config['cols'])]]
    return board

class TetrisAppNoGUI(object):
    def __init__(self):
        self.width = config['cell_size']*config['cols']
        self.height = config['cell_size']*config['rows']
        self.actions = []
        self.needs_actions = True

        self.init_game()

    '''
    Creates random shape
    places shape at the top of the boar din the middle
    checks if collision - if yes game over
    '''  
    def new_stone(self):
        # every stone increases score (fitness)
        self.score += 1

        self.stone = tetris_shapes[rand(len(tetris_shapes))]
        self.stone_x = int(config['cols'] / 2 - len(self.stone[0])/2)
        self.stone_y = 0
        
        if check_collision(self.board,
                        self.stone,
                        (self.stone_x, self.stone_y)):
            self.gameover = True

    '''
        Starts a new game
        - new board
        - new stone
    '''  
    def init_game(self):
        self.score = 0
        self.board = new_board()
        self.new_stone()
        return


    # move a piece horizontally
    def move(self, delta_x):
        if not self.gameover and not self.paused:
            new_x = self.stone_x + delta_x
            if new_x < 0:
                new_x = 0
            if new_x > config['cols'] - len(self.stone[0]):
                new_x = config['cols'] - len(self.stone[0])
            if not check_collision(self.board,
                                    self.stone,
                                    (new_x, self.stone_y)):
                self.stone_x = new_x

    '''
        Try moving piece down
        - if collision:
            1. add stone to board
            2. create new piece
            3. Check for row completion
    ''' 
    def dropFast(self):
        if not self.gameover and not self.paused:
            while not check_collision(self.board, self.stone, (self.stone_x, self.stone_y+1)):
                self.stone_y += 1
            self.board = join_matrixes(self.board, self.stone, (self.stone_x, self.stone_y+1))
            self.new_stone()
            self.needs_actions = True

            while True:
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.score += 10
                            self.board = remove_row(
                            self.board, i)
                            break
                    else:
                        break
    
    def dropNormal(self):
        if not self.gameover and not self.paused:
            self.stone_y += 1
            if check_collision(self.board,
                            self.stone,
                            (self.stone_x, self.stone_y)):
                self.board = join_matrixes(
                    self.board,
                    self.stone,
                    (self.stone_x, self.stone_y))
                self.new_stone()
                self.needs_actions = True
            while True:
                for i, row in enumerate(self.board[:-1]):
                    if 0 not in row:
                        self.score += 10
                        self.board = remove_row(
                        self.board, i)
                        break
                else:
                    break
    
    '''
        Rotate stone if no collision
    ''' 
    def rotate_stone(self):
        if not self.gameover and not self.paused:
            new_stone = rotate_clockwise(self.stone)
            if not check_collision(self.board,
                                    new_stone,
                                    (self.stone_x, self.stone_y)):
                self.stone = new_stone
    
    # pause game 
    def toggle_pause(self):
        self.paused = not self.paused
    
    # start new game if gameover 
    def start_game(self):
        if self.gameover:
            self.init_game()
            self.gameover = False
    
    '''
        Runs the game
        setup game mechanics
    ''' 
    def init(self):
        # controls
        self.key_actions = {
            'LEFT':    lambda:self.move(-1),
            'RIGHT':  lambda:self.move(+1),
            'DOWN':    self.dropFast,
            'UP':    self.rotate_stone,
            'p':    self.toggle_pause,
            'SPACE':  self.start_game
        }
        
        self.gameover = False
        self.paused = False


    def tick(self):
        if self.gameover:
            self.needs_actions = True
        
        # check for events
        if not self.actions:
            self.dropNormal()
        else:	
            action = self.actions[0].upper()
            
            if action in self.key_actions:
                self.key_actions[action]()
                self.actions.pop(0)


    def get_state(self):
        return {"board": numpy.copy(self.board), 
                "stone": numpy.copy(self.stone),
                "stone_x": self.stone_x,
                "stone_y": self.stone_y,
                "score": self.score,
                "gameover": self.gameover,
                "needs_actions": self.needs_actions}

    def add_actions(self, new_actions):
        self.needs_actions = False
        self.actions = new_actions

if __name__ == '__main__':
    App = TetrisAppNoGUI()
    App.init()
    for i in range(30):
        print(App.get_state())
        App.tick()
