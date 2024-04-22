import matplotlib.pyplot as plt
import numpy as np
from typing import List
from abc import ABC, abstractmethod
from collections import deque
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import Button
import random
import heapq 

class BadMaze(Exception):
    pass

class Maze:
    def __init__(self, layout):
        self.layout = layout
        self.start = None
        self.end = None
        self.num_rows = len(layout)
        self.num_cols = len(layout[0])
        
    def is_valid_cell(self, row, col):
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols and self.layout[row][col] == 0

    def get_neighbors(self, node):
        row, col = node.coords
        neighbors = []
        for d_row, d_col in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row, new_col = row + d_row, col + d_col
            if self.is_valid_cell(new_row, new_col):
                neighbors.append(Node((new_row, new_col)))
        return neighbors

    def set_start(self, coords):
        if self.is_valid_cell(coords[0], coords[1]):
            self.start = Node(coords)
        else:
            raise BadMaze("Invalid start coordinates.")

    def set_end(self, coords):
        if self.is_valid_cell(coords[0], coords[1]):
            self.end = Node(coords)
        else:
            raise BadMaze("Invalid end coordinates.")
        
    def is_end(self, node) -> bool:
        return node.coords == self.end.coords
        
    def visualize(self, steps: List['Step'], search_algorithm: 'SearchAlgorithm', final_path: List['Node']):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.layout, cmap='coolwarm')

        if self.start:
            ax.plot(self.start.coords[1], self.start.coords[0], 'go', markersize=10)

        if self.end:
            ax.plot(self.end.coords[1], self.end.coords[0], 'ro', markersize=10)

        step_index = 0
        show_final_path = False
        show_all_steps = False
        show_steps = True
        heatmap_data = np.full((self.num_rows, self.num_cols), np.inf)

        def on_prev_click(event):
            nonlocal step_index
            step_index = (step_index - 1) % len(steps)
            update_plot()

        def on_next_click(event):
            nonlocal step_index
            step_index = (step_index + 1) % len(steps)
            update_plot()

        def on_final_path_clicked(label):
            nonlocal show_final_path
            show_final_path = not show_final_path
            update_plot()

        def on_all_steps_clicked(label):
            nonlocal show_all_steps
            show_all_steps = not show_all_steps
            update_plot()

        def on_steps_clicked(label):
            nonlocal show_steps
            show_steps = not show_steps
            update_plot()

        def animate():
            nonlocal step_index
            for i in range(len(steps)):
                step_index = i
                update_plot()
                fig.canvas.draw()
                fig.canvas.flush_events()
            

        def update_plot():
            ax.clear()
            ax.imshow(self.layout, cmap='YlOrRd')

            if self.start:
                ax.plot(self.start.coords[1], self.start.coords[0], 'go', markersize=10)

            if self.end:
                ax.plot(self.end.coords[1], self.end.coords[0], 'ro', markersize=10)

            if show_all_steps:
                for step in steps:
                    node = step.current_node
                    row, col = node.coords
                    path_cost = step.path_cost
                    heatmap_data[row, col] = min(heatmap_data[row, col], path_cost)

                ax.imshow(heatmap_data, cmap='coolwarm', alpha=0.7)
                for i in range(self.num_rows):
                    for j in range(self.num_cols):
                        if heatmap_data[i, j] != np.inf:
                            ax.text(j, i, int(heatmap_data[i, j]), ha='center', va='center', color='black', fontsize=8)

        
            if show_steps:
                for i in range(step_index + 1):
                    if i < len(steps):
                        current_step = steps[i]
                        current_node = current_step.current_node
                        parent_node = current_step.parent_node

                        if parent_node is not None:
                            ax.arrow(parent_node.coords[1], parent_node.coords[0],
                                    current_node.coords[1] - parent_node.coords[1],
                                    current_node.coords[0] - parent_node.coords[0],
                                    head_width=0.2, head_length=0.2, fc='black', ec='black')
                            

                            #maze.layout[current_node.coords[0]][current_node.coords[1]] = 0.95

            if show_final_path:
                for i in range(len(final_path) - 1):
                    start_node = final_path[i]
                    end_node = final_path[i + 1]
                    ax.arrow(start_node.coords[1], start_node.coords[0], end_node.coords[1] - start_node.coords[1],
                             end_node.coords[0] - start_node.coords[0], head_width=0.1, head_length=0.2,
                             fc='green', ec='green')
                    
            step = steps[step_index]
            node = step.current_node
            ax.plot(node.coords[1], node.coords[0], 'o', color='orange', markersize=10)
            ax.text(node.coords[1], node.coords[0], str(step.path_cost), color='white', ha='center', va='center')

            ax.text(0.05, 0.95, f"Step: {step_index + 1}/{len(steps)}", transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_title(f"Search Algorithm: {search_algorithm.__class__.__name__}")
            ax.axis('off')
            fig.canvas.draw()

        # Create buttons for previous and next steps
        plt.subplots_adjust(bottom=0.2)
        ax_prev = plt.axes([0.4, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.5, 0.05, 0.1, 0.075])
        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')
        btn_prev.on_clicked(on_prev_click)
        btn_next.on_clicked(on_next_click)

        # Create button for animation
        ax_animate = plt.axes([0.6, 0.05, 0.1, 0.075])
        btn_animate = Button(ax_animate, 'Animate')
        btn_animate.on_clicked(lambda event: animate())

        # Create checkboxes for toggling final path, all steps, and steps
        plt.subplots_adjust(left=0.2)
        ax_final_path = plt.axes([0.05, 0.9, 0.1, 0.1])
        ax_all_steps = plt.axes([0.05, 0.8, 0.1, 0.1])
        ax_steps = plt.axes([0.05, 0.7, 0.1, 0.1])
        check_final_path = CheckButtons(ax_final_path, ['Final Path'], [False])
        check_all_steps = CheckButtons(ax_all_steps, ['All Steps'], [False])
        check_steps = CheckButtons(ax_steps, ['Steps'], [True])
        check_final_path.on_clicked(on_final_path_clicked)
        check_all_steps.on_clicked(on_all_steps_clicked)
        check_steps.on_clicked(on_steps_clicked)

        update_plot()
        plt.show()

class Node:
    def __init__(self, coords):
        self.coords = coords

    def __repr__(self):
        return f"Node({self.coords})"
    
    def __hash__(self):
        return hash(self.coords)
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.coords == other.coords
        return False
    
class Step:
    def __init__(self, current_node: Node, parent_node: Node, path_cost: int):
        self.current_node = current_node
        self.parent_node = parent_node
        self.path_cost = path_cost

class SearchAlgorithm(ABC):
    @abstractmethod
    def search(self, maze: Maze) -> tuple[List[Step], List[Node]]:
        pass

    @abstractmethod
    def calculate_path_cost(self, steps: List[Step]) -> int:
        pass
    
class BreadthFirstSearch(SearchAlgorithm):
    def search(self, maze: Maze) -> tuple[List[Step], List[Node]]:
        start_node = maze.start
        end_node = maze.end

        queue = deque([(start_node, None, 0, [start_node])])  # (node, parent, path_cost, path)
        visited = {}  # Store the shortest path cost to each visited node
        steps = []
        best_path = None
        best_path_cost = float('inf')

        while queue:
            current_node, parent_node, path_cost, path = queue.popleft()
            steps.append(Step(current_node, parent_node, path_cost))

            if current_node == end_node:
                if path_cost < best_path_cost:
                    best_path = path
                    best_path_cost = path_cost
            else:
                neighbors = maze.get_neighbors(current_node)
                for neighbor in neighbors:
                    new_path_cost = path_cost + 1
                    if neighbor not in visited or new_path_cost < visited[neighbor]:
                        visited[neighbor] = new_path_cost
                        new_path = path + [neighbor]
                        queue.append((neighbor, current_node, new_path_cost, new_path))

        return steps, best_path

    def calculate_path_cost(self, steps: List[Step]) -> int:
        return steps[-1].path_cost
    
class DepthFirstSearch(SearchAlgorithm):
    def search(self, maze: Maze) -> tuple[List[Step], List[Node]]:
        start_node = maze.start
        end_node = maze.end

        stack = [(start_node, None, 0, [start_node])]  # (node, parent, path_cost, path)
        visited = {}  # Store the shortest path cost to each visited node
        steps = []
        best_path = None
        best_path_cost = float('inf')

        while stack:
            current_node, parent_node, path_cost, path = stack.pop()
            steps.append(Step(current_node, parent_node, path_cost))

            if current_node == end_node:
                if path_cost < best_path_cost:
                    best_path = path
                    best_path_cost = path_cost
            else:
                neighbors = maze.get_neighbors(current_node)
                for neighbor in reversed(neighbors):
                    new_path_cost = path_cost + 1
                    if neighbor not in visited or new_path_cost < visited[neighbor]:
                        visited[neighbor] = new_path_cost
                        new_path = path + [neighbor]
                        stack.append((neighbor, current_node, new_path_cost, new_path))

        return steps, best_path

    def calculate_path_cost(self, steps: List[Step]) -> int:
        return steps[-1].path_cost

def generate_random_maze(rows, cols, obstacle_ratio=0.5):
    layout = [[0] * cols for _ in range(rows)]

    num_obstacles = int(rows * cols * obstacle_ratio)

    for _ in range(num_obstacles):
        while True:
            row = random.randint(0, rows - 1)
            col = random.randint(0, cols - 1)
            if layout[row][col] == 0:
                layout[row][col] = 1
                break
    
    return layout

def test_layout():
    layout = [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0]
        ]
    return layout

use_test_layout = False
maze_x = 10
maze_y = 30
obs = 0.4
search = DepthFirstSearch()

while True:

    try:
        # Define the maze layout
        
        if use_test_layout: 
            layout = test_layout()

        else: 
            layout = generate_random_maze(maze_y, maze_x, obs)

        # Create the maze object
        maze = Maze(layout)

        if use_test_layout:
            maze.set_end((4, 5))
        else: 
            maze.set_end((maze_y-1, maze_x-1))

        # Set the start and end nodes
        maze.set_start((0, 0))

        # Create an instance of the BreadthFirstSearch class
        
        # Perform the search
        print("Starting search...")
        steps, final_path = search.search(maze)
        print("Search completed.")

        if final_path == None: 
            raise BadMaze("No path found.")

        # Visualize the search process
        print("Visualizing the search process...")
        maze.visualize(steps, search, final_path)
        print("Visualization completed.")

        exit()
    
    except BadMaze as e:
        print(e)
        continue

