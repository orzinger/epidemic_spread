import matplotlib.animation as animation
import itertools
import random
import numpy as np
import timeit
from matplotlib import colors
import sys
from matplotlib import pyplot as plt

class Cell:

    def __init__(self):

        self.catch = False

        self.infected = False

    def _update_state(self, obj = None):

        self.catch = not self.catch
        if self.catch:
            self.object = obj
        else:
            self.object = None

    def _is_catched(self):

        return self.catch



class Creature:

    def __init__(self):

        self.infected = False

    def _update_infected(self):

        self.infected = True

    def _is_infected(self):

        return self.infected

class Grid:

    def __init__(self, cells, creature, prob):

        self.grid = None

        self.board_size = cells * cells

        self.creatures = creature

        self.cells_size = cells

        self.board_state = None

        self.prob = prob

        self.init = None

        self.creatures_list = list()

        self.grid_map = None

    def _initGrid(self, outbreak = 0):

        self.grid = [[Cell() for i in range(self.cells_size)] for j in range(self.cells_size)]

        index_1 = random.sample(range(self.cells_size), self.cells_size)

        index_2 = random.sample(range(self.cells_size), self.cells_size)

        initial_grid = list(itertools.product(index_1, index_2))

        initial_grid = random.sample(initial_grid, self.creatures)

        self.grid_map = np.zeros((self.cells_size, self.cells_size))

        for cell in initial_grid:

            i, j = cell

            self.grid[i][j]._update_state()

        self.creatures_list = [Creature() for i in range(self.creatures)]

        self.board_state = initial_grid

        if outbreak == 0:

            cell = random.choice(initial_grid)
            
            self.creatures_list[initial_grid.index(cell)]._update_infected()

            self.grid_map[cell[0]][cell[1]] = 255

            self.init = cell

    def _cell_move(self, k, cell):

        cell_i, cell_j = cell
        cells = [
            (cell_i,cell_j),
            ((cell_i + 1)%self.cells_size, cell_j),
            ((cell_i - 1)%self.cells_size, cell_j),
            (cell_i, (cell_j + 1)%self.cells_size),
            (cell_i, (cell_j - 1)%self.cells_size),
            ((cell_i + 1)%self.cells_size, (cell_j + 1)%self.cells_size),
            ((cell_i - 1)%self.cells_size, (cell_j + 1)%self.cells_size),
            ((cell_i + 1)%self.cells_size, (cell_j - 1)%self.cells_size),
            ((cell_i - 1)%self.cells_size, (cell_j - 1)%self.cells_size)
        ]
        return random.choice(cells), random.sample(cells, k)

    def _cell_neighbors(self, cell, isolated_set):

        cell_i, cell_j = cell
        cells = set([
            ((cell_i + 1)%self.cells_size, cell_j),
            ((cell_i - 1)%self.cells_size, cell_j),
            (cell_i, (cell_j + 1)%self.cells_size),
            (cell_i, (cell_j - 1)%self.cells_size),
            ((cell_i + 1)%self.cells_size, (cell_j + 1)%self.cells_size),
            ((cell_i - 1)%self.cells_size, (cell_j + 1)%self.cells_size),
            ((cell_i + 1)%self.cells_size, (cell_j - 1)%self.cells_size),
            ((cell_i - 1)%self.cells_size, (cell_j - 1)%self.cells_size)
        ])
        return cells.difference(isolated_set)


    # run simulator
    def _run(self, k, n, outbreak = -1):

        count = 1

        counts = list()

        def animate(i):

            nonlocal count

            nonlocal counts

            infected_cells = list()
            
            current_board = []

            new_grid_map = self.grid_map.copy()

            # initial if infection start later
            if i == outbreak:

                cell = random.choice(self.board_state)
            
                self.creatures_list[self.board_state.index(cell)]._update_infected()

                self.grid_map[cell[0]][cell[1]] = 255
            # try to infect cells
            for index, cell in enumerate(self.board_state):

                move, isolated_cells = self._cell_move(k, cell)

                if self.creatures_list[index]._is_infected():

                    neighbors = self._cell_neighbors(cell, isolated_cells)

                    for neighbor in neighbors:

                        if self.grid[neighbor[0]][neighbor[1]]._is_catched():
                        
                            if random.random() <= self.prob:
                                
                                infected_cells.append(self.board_state.index(neighbor))
                                
                                new_grid_map[neighbor[0]][neighbor[1]] = 255

                current_board.append(move)

            x_current_board = []


            # move viruses
            for move, cell in zip(current_board, self.board_state):

                if not self.grid[move[0]][move[1]]._is_catched():

                    self.grid[cell[0]][cell[1]]._update_state()

                    self.grid[move[0]][move[1]]._update_state()

                    x_current_board.append(move)

                else:

                    x_current_board.append(cell)

            self.board_state = x_current_board

            for index in infected_cells:

                if not self.creatures_list[index]._is_infected():
                    
                    self.creatures_list[index]._update_infected()

                    count += 1
            
            counts.append(count)

            mat.set_data(new_grid_map)

            self.grid_map = new_grid_map

            return [mat]

        fig, ax = plt.subplots()

        cmap = colors.ListedColormap(['white', 'black'])

        mat = ax.imshow(self.grid_map, cmap = cmap, extent = [0,self.cells_size,0,self.cells_size])

        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

        plt.title("covid-19 spread\nk:{} p:{} N:{}".format(k, self.prob, n))

        anim = animation.FuncAnimation(fig, animate, frames = n, interval=300, repeat = False, cache_frame_data = False) 

        anim.save("spread_covid_19_k{}_n{}_p{}.gif".format(k,n,self.prob), writer='imagemagick')

        plt.clf()

        return counts









if __name__ == "__main__":

    K = int(sys.argv[2])
    N = int(sys.argv[3])
    G = int(sys.argv[1])

    counts = list()

    if len(sys.argv) > 4:

        p = float(input("enter probability: "))
        k1 = int(input("enter k1: "))
        k2 = int(input("enter k2: "))

        grid = Grid(200, N, p)

        grid._initGrid()

        counts.append(grid._run(k1, G))

        grid = Grid(200, N, p)

        grid._initGrid()

        counts.append(grid._run(k2, G))

        plt.plot(range(G+1), counts[0], label = "k={}".format(k1))

        plt.plot(range(G+1), counts[1], label = "k={}".format(k2))

    else:

        start = timeit.default_timer()

        for p in np.linspace(0.1,1,10):
            
            grid = Grid(200, N, p)

            grid._initGrid()

            counts.append(grid._run(K, G))

        stop = timeit.default_timer()

        print('Time: ', stop - start)

        for i,p in zip(counts, np.linspace(0.1,1,10)):
            
            plt.plot(range(G+1), i, label = "p={}".format(p))


    avg_cells = sum([max(c) for c in counts]) / len(counts)


    plt.ylabel("infected")
    plt.xlim(0,G)
    plt.xlabel("generations")
    plt.title("N:{}".format(N))
    plt.legend()
    plt.figtext(0.8, 0.86,"mean={}".format(avg_cells))
    plt.savefig("model.png")



