from typing import List
from abc import ABC, abstractmethod
from collections import deque
import random

class BadMaze(Exception):
    pass

class Maze:
    def __init__(self, layout):
        
        # Representation of the maze nodes.
        self.nodes = []
        for i in range(len(layout)):
            row = []
            for j in range(len(layout[0])):
                row.append(Node((i, j), layout[i][j] == 0))
            self.nodes.append(row)
        self.start = None
        self.end = None
        self.num_rows = len(layout)
        self.num_cols = len(layout[0])
        self.base_layout = layout

    def get_node(self, coords):
        row, col = coords
        if 0 <= row < self.num_rows and 0 <= col < self.num_cols:
            return self.nodes[row][col]
        return None

    def get_neighbors(self, node):
        row, col = node.coords
        neighbors = []
        for d_row, d_col in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row, new_col = row + d_row, col + d_col
            neighbor = self.get_node((new_row, new_col))
            if neighbor and neighbor.is_passable:
                neighbors.append(neighbor)
        return neighbors

    def set_start(self, coords):
        node = self.get_node(coords)
        if node and node.is_passable:
            self.start = node
        else:
            raise BadMaze("Invalid start coordinates.")

    def set_end(self, coords):
        node = self.get_node(coords)
        if node and node.is_passable:
            self.end = node
        else:
            raise BadMaze("Invalid end coordinates.")
    
    # Success criteria
    def is_end(self, node) -> bool:
        return node.coords == self.end.coords
    
class Node:
    def __init__(self, coords, is_passable):
        self.coords = coords
        self.is_passable = is_passable
        
    def __str__(self):
        return "N"+str(self.coords)
    
    def __hash__(self):
        return hash(self.coords)
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.coords == other.coords and self.is_passable == other.is_passable
            
        return False

class SearchDetails:
    def __init__(self, maze, final_path):
        self.maze = maze
        self.final_path = final_path

    def visualize_maze(self):
        maze = self.maze
        maze_text = [[' '] * maze.num_cols for _ in range(maze.num_rows)]

        # Mark impassable nodes
        for row in range(maze.num_rows):
            for col in range(maze.num_cols):
                if not maze.nodes[row][col].is_passable:
                    maze_text[row][col] = '█'

        # Mark start and end nodes
        start_row, start_col = maze.start.coords
        end_row, end_col = maze.end.coords
        maze_text[start_row][start_col] = 'S'
        maze_text[end_row][end_col] = 'E'

        # Mark final path nodes
        for node in self.final_path:
            if node != maze.start and node != maze.end:
                row, col = node.coords
                maze_text[row][col] = '·'

        # Print the maze representation
        for row in maze_text:
            print(''.join(row))

    def output(self):
        if self.final_path is None:
            print("No path found.")
        else:
            # Visualize the final path
            print("\nVisualization:\n")
            self.visualize_maze()

            # Final path output
            print("\nFinal path:\n")
            final_output = ""
            for node in self.final_path:
                final_output += str(node)

                if node != self.final_path[-1]:
                    final_output += " → "

            print(final_output + "\n")
            
class BreadthFirstSearch():
    def search(self, maze: Maze) -> tuple[List[Node]]:
        start_node = maze.start
        end_node = maze.end
        
        # Initialize the queue with the start node, its parent (None), path cost (0), and path ([start_node])
        queue = deque([(start_node, None, 0, [start_node])])
        
        # Create a dictionary to store the shortest path cost to each visited node
        visited = {start_node: 0}
    
        best_path = None
        
        # Initialize the best path cost to infinity
        best_path_cost = float('inf')

        # Continue the search while the queue is not empty
        while queue:
            
            # Dequeue the next node to explore
            current_node, parent_node, path_cost, path = queue.popleft()
            
            # Check if the current node is the end node
            if current_node == end_node:
                
                # If the current path cost is less than the best path cost, update the best path and its cost
                if path_cost < best_path_cost:
                    best_path = path
                    best_path_cost = path_cost
                    
            else:
                # Get the neighbors of the current node
                neighbors = maze.get_neighbors(current_node)
                
                # Explore each neighbor
                for neighbor in neighbors:
                    
                    # 4. Function to calculate the cost on each path:
                    
                    # Calculate the new path cost by adding 1 to the current path cost
                    new_path_cost = path_cost + 1
                    
                    # Check if the neighbor has not been visited or if the new path cost is less than the previously recorded cost
                    if neighbor not in visited or new_path_cost < visited[neighbor]:
                        
                        # Update the shortest path cost to the neighbor
                        visited[neighbor] = new_path_cost
                        
                        # Create a new path by appending the neighbor to the current path
                        new_path = path + [neighbor]
                        
                        # Path function to accumulate past nodes: Add (the neighbor with its parent, new path cost, and new path) to the queue
                        queue.append((neighbor, current_node, new_path_cost, new_path))
                        
        return SearchDetails(maze, best_path)

def exam_layout():
    layout = [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0]
        ]
    return layout

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

# Ensure this is set to true
use_exam_layout = True

if use_exam_layout:
    
    maze_end = (4, 5)
    layout = exam_layout()
    
else: 
    maze_x = 50
    maze_y = 50
    obs = 0.2
    maze_end = (maze_y-1, maze_x-1)
    layout = generate_random_maze(maze_y, maze_x, obs)

search = BreadthFirstSearch()

while True:

    try:

        # Create the maze object
        maze = Maze(layout)
        maze.set_end(maze_end)
        
        # Set the start and end nodes
        maze.set_start((0, 0))

        # Perform the search
        print("Starting search...")
        search_details = search.search(maze)
        
        # Outputting deails
        print("Search completed:")
        search_details.output()

        exit()
    
    except BadMaze as e:
        print(e)
        continue