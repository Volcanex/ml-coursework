import matplotlib.pyplot as plt
import numpy as np
from typing import List
from abc import ABC, abstractmethod
from collections import deque
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import random
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget, QVBoxLayout, QWidget, QCheckBox, QPushButton
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget, QVBoxLayout, QWidget, QCheckBox, QPushButton
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np


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
        app = QApplication(sys.argv)
        win = QMainWindow()
        win.setWindowTitle('Maze Visualization')

        central_widget = QWidget()
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        win.setCentralWidget(central_widget)

        plot_widget = pg.PlotWidget()
        layout.addWidget(plot_widget)

        view = plot_widget.getViewBox()
        view.setAspectLocked(True)
        view.setRange(pg.QtCore.QRectF(0, 0, self.num_cols, self.num_rows))

        layout_array = np.array(self.layout, dtype=np.float64)  # Convert layout to numpy array
        img = pg.ImageItem(layout_array, border='w')
        view.addItem(img)

        start_node = self.start.coords
        end_node = self.end.coords
        start_marker = pg.ScatterPlotItem(pos=[start_node[::-1]], symbol='o', size=20, pen=None, brush='g')
        end_marker = pg.ScatterPlotItem(pos=[end_node[::-1]], symbol='o', size=20, pen=None, brush='r')
        view.addItem(start_marker)
        view.addItem(end_marker)

        step_index = 0
        show_final_path = False
        show_all_steps = False
        show_steps = True
        heatmap_data = np.full((self.num_rows, self.num_cols), np.inf)

        lines = []
        for step in steps:
            if step.parent_node is not None:
                line = pg.PlotCurveItem(pen=pg.mkPen('y', width=2))
                lines.append(line)
                view.addItem(line)

        final_path_lines = []
        for i in range(len(final_path) - 1):
            line = pg.PlotCurveItem(pen=pg.mkPen('g', width=3))
            final_path_lines.append(line)
            view.addItem(line)

        head_marker = pg.ScatterPlotItem(symbol='o', size=15, pen=None, brush='orange')
        view.addItem(head_marker)

        path_cost_text = pg.TextItem(anchor=(0.5, 0.5))
        view.addItem(path_cost_text)

        def update_plot():
            if show_all_steps:
                for step in steps:
                    node = step.current_node
                    row, col = node.coords
                    path_cost = step.path_cost
                    heatmap_data[row, col] = min(heatmap_data[row, col], path_cost)

                heatmap_data = np.where(heatmap_data == np.inf, np.nan, heatmap_data)  # Replace inf with nan
                img.setImage(heatmap_data, levels=(0, np.nanmax(heatmap_data)))  # Use nanmax to handle nan values

            if show_final_path:
                for i, line in enumerate(final_path_lines):
                    if i < len(final_path) - 1:
                        start_node = final_path[i]
                        end_node = final_path[i + 1]
                        line.setData([start_node.coords[1], end_node.coords[1]], [start_node.coords[0], end_node.coords[0]])

            if show_steps:
                for i, line in enumerate(lines[:step_index]):
                    current_step = steps[i]
                    current_node = current_step.current_node
                    parent_node = current_step.parent_node
                    if parent_node is not None:
                        line.setData([parent_node.coords[1], current_node.coords[1]], [parent_node.coords[0], current_node.coords[0]])

            step = steps[step_index]
            node = step.current_node
            head_marker.setData([node.coords[1]], [node.coords[0]])
            path_cost_text.setText(str(step.path_cost))
            path_cost_text.setPos(node.coords[1], node.coords[0])

        update_plot()

        # Create checkboxes and buttons
        dock_widget = QDockWidget("Controls")
        checkbox_widget = QWidget()
        checkbox_layout = QVBoxLayout()
        checkbox_widget.setLayout(checkbox_layout)

        final_path_checkbox = QCheckBox("Show Final Path")
        final_path_checkbox.stateChanged.connect(lambda state: setattr(self, 'show_final_path', state == Qt.Checked))
        checkbox_layout.addWidget(final_path_checkbox)

        all_steps_checkbox = QCheckBox("Show All Steps")
        all_steps_checkbox.stateChanged.connect(lambda state: setattr(self, 'show_all_steps', state == Qt.Checked))
        checkbox_layout.addWidget(all_steps_checkbox)

        steps_checkbox = QCheckBox("Show Steps")
        steps_checkbox.setChecked(True)
        steps_checkbox.stateChanged.connect(lambda state: setattr(self, 'show_steps', state == Qt.Checked))
        checkbox_layout.addWidget(steps_checkbox)

        prev_button = QPushButton("Previous Step")
        prev_button.clicked.connect(lambda: update_step_index(-1))
        checkbox_layout.addWidget(prev_button)

        next_button = QPushButton("Next Step")
        next_button.clicked.connect(lambda: update_step_index(1))
        checkbox_layout.addWidget(next_button)


        dock_widget.setWidget(checkbox_widget)
        win.addDockWidget(Qt.RightDockWidgetArea, dock_widget)

        def update_step_index(delta):
            nonlocal step_index
            step_index = (step_index + delta) % len(steps)
            update_plot()

        def animate():
            nonlocal step_index
            for i in range(len(steps)):
                step_index = i
                update_plot()
                QApplication.processEvents()  # Process GUI events during animation
                pg.QtGui.QApplication.processEvents()  # Process PyQtGraph events during animation

        win.show()
        sys.exit(app.exec_())
                

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
    
def generate_random_maze(rows, cols, obstacle_ratio=0.4):
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






use_test_layout = True

while True:

    try:
        # Define the maze layout
        
        if use_test_layout: 
            layout = test_layout()
        else: 
            layout = generate_random_maze(40, 40, 0.2)

        # Create the maze object
        maze = Maze(layout)

        if use_test_layout:
            maze.set_end((4, 5))
        else: 
            maze.set_end((39, 39))

        # Set the start and end nodes
        maze.set_start((0, 0))

        # Create an instance of the BreadthFirstSearch class
        bfs = BreadthFirstSearch()

        # Perform the search
        print("Starting search...")
        steps, final_path = bfs.search(maze)
        print("Search completed.")

        # Visualize the search process
        print("Visualizing the search process...")
        maze.visualize(steps, bfs, final_path)
        print("Visualization completed.")

        exit()
    
    except BadMaze as e:
        print(e)
        continue

