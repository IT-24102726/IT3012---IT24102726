from collections import deque
import heapq
import random


class GreedyGridAgent:

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        return random.choice(self.actions_pool)


class SearchAgent:

    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'

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

        else:
            raise ValueError(
                "Invalid algorithm. Use BFS, DFS, or UCS."
            )

    def sense_and_act(self, percept: dict) -> str:

        if not self.plan:

            start = tuple(percept['agent_pos'])

            food_positions = [
                tuple(food)
                for food in percept['all_food']
            ]

            grid_size = percept['grid_size']

            walls = set(
                tuple(wall)
                for wall in percept['walls']
            )

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