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
        
        # 1. Representation of the maze nodes.
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
    
    def visualize_maze_as_text(self, path):
        maze_text = [[' '] * maze.num_cols for _ in range(maze.num_rows)]

        # Mark impassable nodes
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if not maze.nodes[row][col].is_passable:
                    maze_text[row][col] = '█'

        # Mark start and end nodes
        start_row, start_col = self.start.coords
        end_row, end_col = self.end.coords
        maze_text[start_row][start_col] = 'S'
        maze_text[end_row][end_col] = 'E'

        # Mark path nodes
        for node in path:
            if node != self.start and node != self.end:
                row, col = node.coords
                self[row][col] = '·'

        # Print the maze representation
        for row in maze_text:
            print(''.join(row))
        
class Node:
    def __init__(self, coords, is_passable):
        self.coords = coords
        self.is_passable = is_passable

    def __repr__(self):
        return f"Node({self.coords})"
    
    def __hash__(self):
        return hash(self.coords)
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.coords == other.coords
        return False
    
    def __lt__(self, other):
        if isinstance(other, Node):
            return self.coords < other.coords
        return False
    
class Step:
    def __init__(self, current_node: Node, parent_node: Node, path_cost: int, path: List[Node]):
        self.current_node = current_node
        self.parent_node = parent_node
        self.path_cost = path_cost
        self.path = path
        
class SearchDetails:
    def __init__(self, maze, steps, visited, final_path):
        self.maze = maze
        self.steps = steps
        self.visited = visited
        self.final_path = final_path

    def output(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.maze.base_layout, cmap='YlOrRd')

        if self.maze.start:
            ax.plot(self.maze.start.coords[1], self.maze.start.coords[0], 'go', markersize=10)

        if self.maze.end:
            ax.plot(self.maze.end.coords[1], self.maze.end.coords[0], 'ro', markersize=10)

        step_index = 0
        show_final_path = False
        show_all_steps = False
        show_steps = True
        heatmap_data = np.full((self.maze.num_rows, self.maze.num_cols), np.inf)

        def on_prev_click(event):
            nonlocal step_index
            step_index = (step_index - 1) % len(self.steps)
            update_plot()

        def on_next_click(event):
            nonlocal step_index
            step_index = (step_index + 1) % len(self.steps)
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
            for i in range(len(self.steps)):
                step_index = i
                update_plot()
                fig.canvas.draw()
                fig.canvas.flush_events()

        def update_plot():
            ax.clear()
            ax.imshow(self.maze.base_layout, cmap='YlOrRd')

            if self.maze.start:
                ax.plot(self.maze.start.coords[1], self.maze.start.coords[0], 'go', markersize=10)

            if self.maze.end:
                ax.plot(self.maze.end.coords[1], self.maze.end.coords[0], 'ro', markersize=10)

            if show_all_steps:
                for node, path_cost in self.visited.items():
                    row, col = node.coords
                    heatmap_data[row, col] = path_cost

                ax.imshow(heatmap_data, cmap='coolwarm', alpha=0.7)
                for i in range(self.maze.num_rows):
                    for j in range(self.maze.num_cols):
                        if heatmap_data[i, j] != np.inf:
                            ax.text(j, i, int(heatmap_data[i, j]), ha='center', va='center', color='black', fontsize=8)

            if show_steps:
                for i in range(step_index + 1):
                    if i < len(self.steps):
                        current_step = self.steps[i]
                        current_node = current_step.current_node
                        parent_node = current_step.parent_node
                        if parent_node.coords == current_node.coords:
                            
                            ax.text(current_node.coords[1], current_node.coords[0], 'X', ha='center', va='center', color='red', fontsize=12)
                            
                        else:
                            if parent_node is not None:
                                ax.arrow(parent_node.coords[1], parent_node.coords[0],
                                        current_node.coords[1] - parent_node.coords[1],
                                        current_node.coords[0] - parent_node.coords[0],
                                        head_width=0.2, head_length=0.2, fc='black', ec='black', alpha=1)
                                    

                        if i == step_index:
                            path = current_step.path
                            for j in range(len(path) - 1):
                                start_node = path[j]
                                end_node = path[j + 1]
                                ax.plot([start_node.coords[1], end_node.coords[1]],
                                        [start_node.coords[0], end_node.coords[0]],
                                        color='cyan', linewidth=2, alpha=0.8)

            if show_final_path:
                for i in range(len(self.final_path) - 1):
                    start_node = self.final_path[i]
                    end_node = self.final_path[i + 1]
                    ax.arrow(start_node.coords[1], start_node.coords[0],
                             end_node.coords[1] - start_node.coords[1],
                             end_node.coords[0] - start_node.coords[0],
                             head_width=0.1, head_length=0.2, fc='green', ec='green')

            step = self.steps[step_index]
            node = step.current_node
            ax.plot(node.coords[1], node.coords[0], 'o', color='orange', markersize=10)
            ax.text(node.coords[1], node.coords[0], str(step.path_cost), color='white', ha='center', va='center')

            ax.text(0.05, 0.95, f"Step: {step_index + 1}/{len(self.steps)}", transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            #ax.set_title(f"Search Algorithm: {search_algorithm.__class__.__name__}")
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
        
class SearchAlgorithm(ABC):
    @abstractmethod
    def search(self, maze: Maze) -> tuple[List[Step], List[Node]]:
        pass

class BreadthFirstSearch(SearchAlgorithm):
    def search(self, maze: Maze) -> SearchDetails:
        start_node = maze.start
        end_node = maze.end

        visited = {start_node: 0} 
        queue = deque([Step(start_node, None, 0, [start_node])])  
        all_steps = [] 
        best_path = None
        best_path_cost = float('inf')

        while queue:
            current_step = queue.popleft()
            current_node = current_step.current_node
            path_cost = current_step.path_cost
            path = current_step.path

            all_steps.append(current_step)

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
                        new_step = Step(neighbor, current_node, new_path_cost, new_path)
                        queue.append(new_step)

        return SearchDetails(maze, all_steps, visited, best_path)

class ForwardChaining(SearchAlgorithm):
    def search(self, maze):
        # Fact types
        # Known: (Y, X), (Bool)
        # CurrentNode: (Y, X), (Y', X')  # Parent node is for visuals only
        # VisitedNode: (Y, X) , (Index)
        # EndNode: (Y, X)
        # StartNode: (Y, X)
        # ParentChild: (Y, X), (Y', X')
        # PreviousNode: (Y, X)
        # CanMove: (Y, X)
        
        start_node = maze.start.coords
        end_node = maze.end.coords
        
        facts: list[tuple] = [
            ("Known", (start_node, True)),
            ("CurrentNode", start_node, start_node),
            ("EndNode", end_node),
            ("StartNode", start_node),
            ("VisitedNode", start_node, 0),
            ("ParentChild", start_node, start_node),
        ]

        def sensor(maze: Maze, facts: list[tuple]) -> list[tuple]:
            current_node_coords = next((node for item in facts if isinstance(item, tuple) and len(item) == 3 and item[0] == "CurrentNode" for node in [item[1]]), None)
            if current_node_coords is None:
                return facts  

            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, Down, Right, Left

            for dy, dx in directions:
                y, x = current_node_coords
                neighbor_coords = (y + dy, x + dx)

                if 0 <= neighbor_coords[0] < maze.num_rows and 0 <= neighbor_coords[1] < maze.num_cols:
                    node = maze.nodes[neighbor_coords[0]][neighbor_coords[1]]

                    if ("Known", (neighbor_coords, node.is_passable)) not in facts:
                        facts.append(("Known", (neighbor_coords, node.is_passable)))
                        print("Detected node:")
                        print(neighbor_coords)
                else:

                    if ("Known", (neighbor_coords, False)) not in facts:
                        facts.append(("Known", (neighbor_coords, False)))
                        print("Detected node (Outside Maze):")
                        print(neighbor_coords)

            return facts
        
        def rule_movement_restrictions(facts: list[tuple]) -> list[tuple]:
            current_node_coords = next((node for item in facts if isinstance(item, tuple) and len(item) == 3 and item[0] == "CurrentNode" for node in [item[1]]), None)
            if current_node_coords is None:
                return facts

            # Delete existing CanMove facts from the knowledge base
            facts = [fact for fact in facts if fact[0] != "CanMove"]

            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, Down, Right, Left

            for dy, dx in directions:
                y, x = current_node_coords
                neighbor_coords = (y + dy, x + dx)

                known_fact = next((fact for fact in facts if isinstance(fact, tuple) and fact[0] == "Known" and fact[1][0] == neighbor_coords), None)
                if known_fact:
                    _, (_, is_passable) = known_fact
                    if is_passable:
                        facts.append(("CanMove", (current_node_coords, neighbor_coords)))

            return facts
        
        def rule_make_parent_child(facts: list[tuple]) -> list[tuple]:
            current_node = next((fact[1:] for fact in facts if fact[0] == "CurrentNode"), None)
            if current_node is None:
                return facts

            current_node_coords, previous_node_coords = current_node
            visited_nodes = [fact[1] for fact in facts if fact[0] == "VisitedNode"]
            parent_child_nodes = set((fact[1], fact[2]) for fact in facts if fact[0] == "ParentChild")

            for node in visited_nodes:
                if node not in [parent_child[1] for parent_child in parent_child_nodes]:
                    if abs(node[0] - previous_node_coords[0]) + abs(node[1] - previous_node_coords[1]) == 1:
                        facts.append(("ParentChild", previous_node_coords, node))

            return facts
        
        
        def rule_progress_or_backtrack(facts: list[tuple]) -> list[tuple]:
            current_node = next((node for item in facts if isinstance(item, tuple) and len(item) == 3 and item[0] == "CurrentNode" for node in [item[1:3]]), None)
            print(f"Current node: {current_node}")
            if current_node:
                current_node_coords, previous_node_coords = current_node
                traversable_nodes = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    new_node = (current_node_coords[0] + dy, current_node_coords[1] + dx)
                    if any(item[0] == "Known" and item[1] == (new_node, True) for item in facts):
                        traversable_nodes.append(new_node)
                print(f"Traversable nodes: {traversable_nodes}")

                if traversable_nodes:
                    # Check if there's a "VisitedNode" fact and a "CanMove" fact for each traversable node
                    for index, next_node in enumerate(traversable_nodes):
                        visited_node_fact = next((fact for fact in facts if isinstance(fact, tuple) and fact[0] == "VisitedNode" and fact[1] == next_node), None)
                        can_move_fact = next((fact for fact in facts if isinstance(fact, tuple) and fact[0] == "CanMove" and fact[1][1] == next_node), None)
                        if not visited_node_fact and can_move_fact:

                            # Select the next node to visit
                            print(f"Next node: {next_node}")
                            facts = [item for item in facts if not isinstance(item, tuple) or item[0] != "CurrentNode"]
                            facts.append(("CurrentNode", next_node, current_node_coords))  # Store the previous node
                            return facts

                    # Backtrack to the previously visited node
                    current_node_visited_fact = next((fact for fact in facts if isinstance(fact, tuple) and fact[0] == "VisitedNode" and fact[1] == current_node_coords), None)
                    if current_node_visited_fact:
                        current_node_visited_index = current_node_visited_fact[2]
                    else:
                        current_node_visited_index = 0

                    traversable_and_visited = []
                    for traversable_node in traversable_nodes:
                        visited_node_fact = next((fact for fact in facts if isinstance(fact, tuple) and fact[0] == "VisitedNode" and fact[1] == traversable_node), None)
                        if visited_node_fact:
                            traversable_and_visited.append((traversable_node, visited_node_fact[2]))
                        else:
                            traversable_and_visited.append((traversable_node, float('inf')))

                    print(f"Traversable and visited: {traversable_and_visited}")

                    traversable_and_visited = [node for node in traversable_and_visited if node[1] <= current_node_visited_index]
                    traversable_and_visited.sort(key=lambda x: x[1], reverse=True)

                    if traversable_and_visited:
                        next_node = traversable_and_visited[0][0]
                        print(f"Next node: {next_node}")
                        facts = [item for item in facts if not isinstance(item, tuple) or item[0] != "CurrentNode"]
                        facts.append(("CurrentNode", next_node, current_node_coords))  # Store the previous node
                        print(f"Updated facts: {facts}")
                        return facts

            print(f"No changes made to facts: {facts}")
            return facts
        
        def rule_visited(facts: list[tuple]) -> list[tuple]:
            # Add the Visited fact for the current node
            current_node = next((fact[1] for fact in facts if fact[0] == "CurrentNode"), None)
            if current_node:
                visited_facts = [fact for fact in facts if fact[0] == "VisitedNode" and fact[1] == current_node]
                if not visited_facts:
                    visited_index = max([fact[2] for fact in facts if fact[0] == "VisitedNode"] + [0])
                    facts.append(("VisitedNode", current_node, visited_index + 1))
                return facts

            return facts

        def end_process():
            # Replace the button with a label
            end_label = plt.text(-1.1, 0.5, "", fontsize=12, ha='center', va='center')

            # Check if there's a "VisitedNode" fact with the same coords as the end node fact
            end_node = next((fact[1] for fact in facts if fact[0] == "EndNode"), None)
            if end_node is not None:
                end_visited = any(fact[0] == "VisitedNode" and fact[1] == end_node for fact in facts)
                if end_visited:
                    end_label.set_text("End Found!")

                    # Trace back the path from the end node to the start node
                    final_path = []
                    tracer = end_node
                    start_node = next((fact[1] for fact in facts if fact[0] == "StartNode"), None)
                    if start_node is not None:
                        while tracer != start_node:
                            final_path.append(tracer)
                            parent_child_facts = [fact for fact in facts if fact[0] == "ParentChild" and fact[2] == tracer]
                            if parent_child_facts:
                                parent_child_fact = max(parent_child_facts, key=lambda x: facts.index(x))
                                tracer = parent_child_fact[1]
                            else:
                                break
                        final_path.append(start_node)  # Add the start node to the final path

                        # Draw the final path
                        for i in range(len(final_path) - 1):
                            start_node_coords = final_path[i]
                            end_node_coords = final_path[i + 1]
                            ax_maze.arrow(end_node_coords[1], end_node_coords[0],
                                        start_node_coords[1] - end_node_coords[1], start_node_coords[0] - end_node_coords[0],
                                        head_width=0.2, head_length=0.2, fc='green', ec='green')
                else:
                    end_label.set_text("No End Found.")
            else:
                end_label.set_text("No End Node found.")

            fig.canvas.draw()
                
        def update_plot():
            ax_maze.clear()
            ax_kb.clear()

            # Draw the maze
            ax_maze.imshow(maze.base_layout, cmap='YlOrRd')
            
            # Draw fog of war
            known_nodes = [(fact[1][0]) for fact in facts if fact[0] == "Known"]
            for i in range(maze.num_rows):
                for j in range(maze.num_cols):
                    if (i, j) not in known_nodes:
                        ax_maze.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='none', facecolor='black', alpha=0.3))
                        
            for fact in facts:
                if fact[0] == "Known":
                    coords, is_passable = fact[1]
                    i, j = coords
                    if 0 <= i < maze.num_rows and 0 <= j < maze.num_cols:
                        if is_passable:
                            ax_maze.text(j, i, f"Known", ha='center', va='bottom', color='black', fontsize=8)
                        else:
                            ax_maze.text(j, i, f"Known\nImpassable", ha='center', va='center', color='red', fontsize=8)
                    else:
                        ax_maze.text(j, i, f"Known\nBoundary", ha='center', va='center', color='darkred', fontsize=8)

                    ax_maze.text(j-0.3, i-0.3, f"{i},{j}", ha='center', va='center', color='black', fontsize=6)

                if fact[0] == "VisitedNode":
                    i, j = fact[1]
                    ax_maze.text(j, i, f"Visited({str(fact[2])})", ha='center', va='top', color='purple', fontsize=8)
            
            # Mark start and end nodes
            start_row, start_col = maze.start.coords
            end_row, end_col = maze.end.coords
            ax_maze.plot(start_col, start_row, 'go', markersize=10)
            ax_maze.plot(end_col, end_row, 'ro', markersize=10)

            # Mark current node
            current_node = next((fact[1:] for fact in facts if fact[0] == "CurrentNode"), None)
            if current_node:
                current_node_coords, previous_node_coords = current_node
                row, col = current_node_coords
                ax_maze.plot(col, row, 'bo', markersize=10)

            # Display knowledge base
            fact_strings = []
            max_length = 15
            for fact in facts:
                if fact[0] == "Known":
                    fact_string = f"(Known: {str(fact[1][0][0])},{str(fact[1][0][1])},{str(fact[1][1])})"
                elif fact[0] == "VisitedNode":
                    fact_string = f"(VisitedNode: {str(fact[1])},N={str(fact[2])})"
                elif fact[0] == "ParentChild":
                    fact_string = f"(ParentChild: P={str(fact[1])}, C={str(fact[2])})"
                else:
                    fact_string = f"({fact[0]}: {str(fact[1][0])},{str(fact[1][1])})"
                
                if len(fact_string) < max_length:
                    fact_string += (max_length - len(fact_string)) * " "
                
                fact_strings.append(fact_string)
            
            kb_text = "\n".join(fact_strings)
            ax_kb.text(0, 0, kb_text, fontsize=6)

            visited_facts = [fact for fact in facts if fact[0] == "VisitedNode"]
            visited_coords = [fact[1] for fact in visited_facts]
            
            # Draw arrows using the "ParentChild" facts
            parent_child_facts = [(fact[1], fact[2]) for fact in facts if fact[0] == "ParentChild"]
            for parent_coords, child_coords in parent_child_facts:
                if child_coords in visited_coords:
                    ax_maze.arrow(parent_coords[1], parent_coords[0],
                                child_coords[1] - parent_coords[1], child_coords[0] - parent_coords[0],
                                head_width=0.2, head_length=0.2, fc='navy', ec='navy', alpha=0.5)
            
            # Draw red arrow from previous "CurrentNode" to current "CurrentNode"
            if current_node and previous_node_coords != current_node_coords:
                ax_maze.arrow(previous_node_coords[1], previous_node_coords[0],
                            current_node_coords[1] - previous_node_coords[1], current_node_coords[0] - previous_node_coords[0],
                            head_width=0.2, head_length=0.2, fc='red', ec='red', alpha=1)
            
            ax_maze.set_title("Maze")
            ax_maze.axis('off')
            ax_kb.set_title("Knowledge Base")
            ax_kb.axis('off')

            fig.canvas.draw()

        rules = (rule_movement_restrictions, rule_make_parent_child, rule_progress_or_backtrack, 
        rule_visited)
        old_facts = [None]
        
        fig, (ax_maze, ax_kb) = plt.subplots(1, 2, figsize=(12, 6))
        fig.subplots_adjust(bottom=0.2)

        # Create the "Iterate" button
        ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
        button = Button(ax_button, 'Iterate')


        def on_iterate(event):
            nonlocal old_facts, facts
            if old_facts[-1] != facts:
                old_facts.append(facts.copy())  
                print("\n\nSensor:\n")
                facts = sensor(maze, facts)
                for rule in rules:

                    result = rule(facts)
                    if result is not None:
                        facts = result

                    print(f"\n\nRule: {rules.index(rule)}\n")
                    new_facts = [fact for fact in facts if fact not in old_facts]
                    for fact in new_facts:
                        print(f"{fact[0]} : {fact[1:]}")

                print(f"\n\n-----------------------------------------------------\n")

                print(facts)
                update_plot()
                
            else:
                end_process()
                
            if old_facts[-1] == facts:
                end_process()

        button.on_clicked(on_iterate)

        update_plot()
        plt.show()

        return None
    
class AStarSearch(SearchAlgorithm):
    def search(self, maze: Maze) -> SearchDetails:
        start_node = maze.start
        end_node = maze.end

        # Initialize the open list (priority queue), visited dict, and steps list
        open_list = [(0, start_node)]
        visited = {start_node: 0}
        steps = []
        g_score = {start_node: 0}
        came_from = {start_node: None}

        while open_list:
            _, current_node = heapq.heappop(open_list)

            if current_node == end_node:
                path = self.reconstruct_path(came_from, current_node)
                steps.append(Step(current_node, came_from[current_node], g_score[current_node], path))
                return SearchDetails(maze, steps, visited, path)

            visited[current_node] = g_score[current_node]

            if len(steps) > 0:
                if steps[-1].current_node != current_node:
                    steps.append(Step(current_node, came_from[current_node], g_score[current_node], []))
            
            else:
                steps.append(Step(current_node, came_from[current_node], g_score[current_node], []))

            for neighbor in maze.get_neighbors(current_node):
                tentative_g_score = g_score[current_node] + 1

                if neighbor not in visited or tentative_g_score < visited[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, end_node)
                    heapq.heappush(open_list, (f_score, neighbor))

        return SearchDetails(maze, steps, visited, None)

    def heuristic(self, node: Node, end_node: Node) -> int:
        # Manhattan distance heuristic
        return abs(node.coords[0] - end_node.coords[0]) + abs(node.coords[1] - end_node.coords[1])

    def reconstruct_path(self, came_from: dict, current_node: Node) -> List[Node]:
        path = [current_node]
        while current_node in came_from and came_from[current_node] is not None:
            current_node = came_from[current_node]
            path.append(current_node)
        path.reverse()
        return path
    
def generate_random_maze(rows, cols, obstacle_ratio=1):
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

def exam_layout():
    layout = [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0]
        ]
    return layout

# Ensure this is set to true
use_exam_layout = True
search = ForwardChaining()

while True:
    try:
        
        if use_exam_layout:
            maze_end = (4, 5)
            layout = exam_layout()
        else: 
            maze_x = 10
            maze_y = 10
            obs = 0.25
            maze_end = (maze_y-1, maze_x-1)
            layout = generate_random_maze(maze_y, maze_x, obs)

        # Create the maze object
        maze = Maze(layout)
        maze.set_end(maze_end)
        
        # Set the start and end nodes
        maze.set_start((0, 0))

        # Perform the search
        print("Starting search...")
        search_details = search.search(maze)
        
        if search_details is not None:
            if search_details.final_path == None:
                raise BadMaze("Couldn't find a final path.")
        
            # Outputting deails
            print("Search completed:")
            search_details.output()


        exit()
    
    except BadMaze as e:
        print(e)
        continue
    
