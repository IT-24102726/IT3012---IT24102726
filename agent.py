from collections import deque
import heapq
import random
import math


class GreedyGridAgent:

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        return random.choice(self.actions_pool)


class SearchAgent:

    def __init__(self):
        self.plan = []
        self.active_algo = 'AStar'

    def get_neighbors(self, state, grid_size, walls):
        x, y = state
        width, height = grid_size

        moves = [
            ('Up', (x, y + 1)),
            ('Down', (x, y - 1)),
            ('Left', (x - 1, y)),
            ('Right', (x + 1, y))
        ]

        neighbors = []

        for action, new_state in moves:
            nx, ny = new_state

            if nx < 0 or nx >= width:
                continue

            if ny < 0 or ny >= height:
                continue

            if new_state in walls:
                continue

            neighbors.append((new_state, action))

        return neighbors

    def reconstruct_path(self, parent, action_taken, start, goal):
        actions = []
        current = goal

        while current != start:
            actions.append(action_taken[current])
            current = parent[current]

        actions.reverse()

        return actions

    def manhattan_distance(self, pos, goal):
        return (
            abs(pos[0] - goal[0])
            + abs(pos[1] - goal[1])
        )

    def euclidean_distance(self, pos, goal):
        return math.sqrt(
            (pos[0] - goal[0]) ** 2
            + (pos[1] - goal[1]) ** 2
        )

    def astar_search(
        self,
        start_pos,
        goal_pos,
        walls,
        grid_size,
        heuristic_type='manhattan'
    ):
        frontier = []
        counter = 0

        if heuristic_type == 'manhattan':
            heuristic = self.manhattan_distance
        elif heuristic_type == 'euclidean':
            heuristic = self.euclidean_distance
        else:
            raise ValueError(
                "Invalid heuristic. Use manhattan or euclidean."
            )

        g_cost = 0
        h_cost = heuristic(start_pos, goal_pos)
        f_cost = g_cost + h_cost

        heapq.heappush(
            frontier,
            (
                f_cost,
                g_cost,
                counter,
                start_pos,
                []
            )
        )

        reached_states = set()

        while frontier:
            f_cost, g_cost, _, current_pos, path_taken = heapq.heappop(
                frontier
            )

            if current_pos == goal_pos:
                return path_taken

            if current_pos in reached_states:
                continue

            reached_states.add(current_pos)

            for next_pos, action in self.get_neighbors(
                current_pos,
                grid_size,
                walls
            ):
                if next_pos in reached_states:
                    continue

                new_g_cost = g_cost + 1
                new_h_cost = heuristic(
                    next_pos,
                    goal_pos
                )
                new_f_cost = new_g_cost + new_h_cost

                counter += 1

                heapq.heappush(
                    frontier,
                    (
                        new_f_cost,
                        new_g_cost,
                        counter,
                        next_pos,
                        path_taken + [action]
                    )
                )

        return []

    def bfs_search(self, start, goal, grid_size, walls):
        frontier = deque([start])
        reached = {start}

        parent = {}
        action_taken = {}

        while frontier:
            current = frontier.popleft()

            if current == goal:
                return self.reconstruct_path(
                    parent,
                    action_taken,
                    start,
                    goal
                )

            for next_state, action in self.get_neighbors(
                current,
                grid_size,
                walls
            ):
                if next_state not in reached:
                    reached.add(next_state)

                    parent[next_state] = current
                    action_taken[next_state] = action

                    frontier.append(next_state)

        return []

    def dfs_search(self, start, goal, grid_size, walls):
        frontier = [start]
        reached = {start}

        parent = {}
        action_taken = {}

        while frontier:
            current = frontier.pop()

            if current == goal:
                return self.reconstruct_path(
                    parent,
                    action_taken,
                    start,
                    goal
                )

            for next_state, action in self.get_neighbors(
                current,
                grid_size,
                walls
            ):
                if next_state not in reached:
                    reached.add(next_state)

                    parent[next_state] = current
                    action_taken[next_state] = action

                    frontier.append(next_state)

        return []

    def ucs_search(self, start, goal, grid_size, walls):
        frontier = []
        counter = 0

        heapq.heappush(
            frontier,
            (0, counter, start)
        )

        reached = {start}

        parent = {}
        action_taken = {}

        while frontier:
            cost, _, current = heapq.heappop(frontier)

            if current == goal:
                return self.reconstruct_path(
                    parent,
                    action_taken,
                    start,
                    goal
                )

            for next_state, action in self.get_neighbors(
                current,
                grid_size,
                walls
            ):
                if next_state not in reached:
                    reached.add(next_state)

                    parent[next_state] = current
                    action_taken[next_state] = action

                    counter += 1

                    heapq.heappush(
                        frontier,
                        (
                            cost + 1,
                            counter,
                            next_state
                        )
                    )

        return []

    def search(self, start, goal, grid_size, walls):

        if self.active_algo == 'BFS':
            return self.bfs_search(
                start,
                goal,
                grid_size,
                walls
            )

        elif self.active_algo == 'DFS':
            return self.dfs_search(
                start,
                goal,
                grid_size,
                walls
            )

        elif self.active_algo == 'UCS':
            return self.ucs_search(
                start,
                goal,
                grid_size,
                walls
            )

        elif self.active_algo == 'AStar':
            return self.astar_search(
                start,
                goal,
                walls,
                grid_size,
                heuristic_type='manhattan'
            )

        else:
            raise ValueError(
                "Invalid algorithm. Use BFS, DFS, UCS, or AStar."
            )

    def sense_and_act(self, percept: dict) -> str:

        if not self.plan:

            start = tuple(percept['agent_pos'])

            food_positions = [
                tuple(food)
                for food in percept['all_food']
            ]

            grid_size = percept['grid_size']

            walls = {
                tuple(wall)
                for wall in percept['walls']
            }

            if not food_positions:
                return 'Stay'

            def manhattan_distance(food):
                return (
                    abs(start[0] - food[0])
                    + abs(start[1] - food[1])
                )

            goal = min(
                food_positions,
                key=manhattan_distance
            )

            if self.active_algo == 'AStar':
                self.plan = self.astar_search(
                    start_pos=start,
                    goal_pos=goal,
                    walls=walls,
                    grid_size=grid_size,
                    heuristic_type='manhattan'
                )

            else:
                self.plan = self.search(
                    start,
                    goal,
                    grid_size,
                    walls
                )

        if self.plan:
            return self.plan.pop(0)

        return random.choice(
            ['Up', 'Down', 'Left', 'Right']
        )