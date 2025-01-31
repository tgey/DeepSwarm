# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import math
import random

from . import cfg, left_cost_is_better
from .log import Log
from .nodes import Node, NeighbourNode, ParentNode
from .util import get_size, getsize
import tensorflow as tf

import os
import logging

import time # TODO LOW debug
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL # TODO LOW debug
logging.getLogger('tensorflow').setLevel(logging.FATAL)

class ACO:
    """Class responsible for performing Ant Colony Optimization."""

    def __init__(self, backend, storage):
        self.graph = Graph()
        self.current_depth = 0
        self.backend = backend
        self.storage = storage

    def search(self):
        """Performs neural architecture search using Ant colony optimization.

        Returns:
            ant which found the best network topology.
        """

        # Generate random ant only if the search started from zero
        if not self.storage.loaded_from_save:
            Log.header("STARTING ACO SEARCH", type="GREEN")
            self.best_ant = Ant(self.graph.generate_path(self.random_select))
            self.best_ant.evaluate(self.backend, self.storage)
            Log.info(self.best_ant)
        else:
            Log.header("RESUMING ACO SEARCH", type="GREEN")

        while self.graph.current_depth <= cfg['max_depth']:
            Log.header(f'Current search depth is {self.graph.current_depth}' , type="GREEN")
            ants = self.generate_ants()

            # Sort ants using user selected metric
            ants.sort() if cfg['metrics'] == 'loss' else ants.sort(reverse=True)

            # Update the best ant if new better ant is found
            if left_cost_is_better(ants[0].cost, self.best_ant.cost):
                self.best_ant = ants[0]
                Log.header("NEW BEST ANT FOUND", type="GREEN")

            # Log best ant information
            Log.header("BEST ANT DURING ITERATION")
            Log.info(self.best_ant)

            # Perform global pheromone update
            self.update_pheromone(ant=self.best_ant, update_rule=self.global_update)

            # Print pheromone information and increase the graph's depth
            self.graph.show_pheromone()
            self.graph.increase_depth()

            # Perform a backup
            self.storage.perform_backup()
        
        return self.best_ant

    def generate_ants(self):
        """Generates a new ant population.

        Returns:
            list containing different evaluated ants.
        """

        ants = []
        for ant_number in range(cfg['aco']['ant_count']):
            Log.header(f'GENERATING ANT {ant_number + 1} FOR DEPTH {self.graph.current_depth}')
            ant = Ant()
            # Generate ant's path using ACO selection rule
            ant.path = self.graph.generate_path(self.aco_select)
            # Evaluate how good is the new path
            ant.evaluate(self.backend, self.storage)
            ants.append(ant)
            Log.info(ant)
            # Perform local pheromone update
            self.update_pheromone(ant=ant, update_rule=self.local_update)
        return ants

    def random_select(self, node: Node):
        """Randomly selects one neighbour node and its attributes.

        Args:
            neighbours [NeighbourNode]: list of neighbour nodes.
        Returns:
            a randomly selected neighbour node.
        """
        neighbours = node.neighbours
        random_choice = random.choice(neighbours)
        current_node = self.graph.get_node_by_name_and_depth(random_choice.name, random_choice.depth)
        current_node.select_random_attributes()
        return current_node

    def aco_select(self, node: Node):
        """Selects one neighbour node and its attributes using ACO selection rule.

        Args:
            neighbours [NeighbourNode]: list of neighbour nodes.
        Returns:
            selected neighbour node.
        """
        neighbours = node.neighbours
        # Transform a list of NeighbourNode objects to list of tuples
        # (Node, pheromone, heuristic)
        
        tuple_neighbours = []
        for n in neighbours:
            #Check if neighbour if a direct neighbour or a residual neighbour
            if n.depth - (node.depth + 1) != 0:
                # TODO HIGH benchmark the activation formula
                uniform = (1 / (n.depth - (node.depth + 1)))
                normalized = ((self.graph.current_depth - 1) / (cfg['max_depth'] - 1))
                # Log.debug(f'normalized: {normalized} uniform: {uniform}') # TODO LOW debug
                new_heuristic = n.find_parent(node).heuristic * normalized * uniform
            else:
                new_heuristic = n.find_parent(node).heuristic
            # Log.debug(f'heuristic value: {new_heuristic} for {n.node} with pheromone {n.find_parent(node).pheromone}') # TODO LOW debug
            neigh = self.graph.get_node_by_name_and_depth(n.name, n.depth)
            if (neigh, n.find_parent(node).pheromone, new_heuristic) not in tuple_neighbours:
                tuple_neighbours.append((neigh, n.find_parent(node).pheromone, new_heuristic))
        # Select node using ant colony selection rule
        current_node = self.aco_select_rule(tuple_neighbours)
        # Select custom attributes using ant colony selection rule
        current_node.select_custom_attributes(self.aco_select_rule)
        return current_node

    def aco_select_rule(self, neighbours: list):
        """Selects neigbour using ACO transition rule.

        Args:
            neighbours [(Object, float, float)]: list of tuples, where each tuple
            contains: an object to be selected, object's pheromone value and
            object's heuristic value.
        Returns:
            selected object.
        """

        probabilities = []
        denominator = 0.0
        # Calculate probability for each neighbour
        for (_, pheromone, heuristic) in neighbours: # TODO MED if skip more than 1 layer, need to be optimized
            probability = pheromone * heuristic
            probabilities.append(probability)
            denominator += probability

        # Try to perform greedy select: exploitation
        random_variable = random.uniform(0, 1)
        # greediness_threshold = cfg['aco']['greediness'] * (self.graph.current_depth / cfg['max_depth'])
        # if random_variable <= greediness_threshold:
        if random_variable <= cfg['aco']['greediness']:
            # Log.debug("EXPLOITATION")
            # Find max probability
            max_probability = max(probabilities)
            # Gather the indices of probabilities that are equal to the max probability
            max_indices = [i for i, j in enumerate(probabilities) if j == max_probability]
            # From those max indices select random index
            neighbour_index = random.choice(max_indices)
            return neighbours[neighbour_index][0]

        # Otherwise perform select using roulette wheel: exploration
        # Log.debug("EXPLORATION")
        return (random.choice(neighbours)[0])

    def update_pheromone(self, ant, update_rule):
        """Updates the pheromone using given update rule.

        Args:
            ant: ant which should perform the pheromone update.
            update_rule: function which takes pheromone value and ant's cost,
            and returns a new pheromone value.
        """

        current_node = self.graph.input_node
        # Skip the input node as it's not connected to any previous node
        for node in ant.path[1:]:
            # Use a node from the path to retrieve its corresponding instance from the graph
            neighbour = next((x for x in current_node.neighbours if x.name == node.name and \
                                                                    x.depth == node.depth), None)

            # If the path was closed using complete_path method, ignore the rest of the path
            if neighbour is None:
                break

            # Update pheromone connecting to a neighbour
            parent_node = neighbour.find_parent(current_node) 
            parent_node.pheromone = update_rule(
                old_value=parent_node.pheromone,
                cost=ant.cost
            )

            # Update attribute's pheromone values
            neigh = self.graph.get_node_by_name_and_depth(neighbour.name, neighbour.depth)
            for attribute in neigh.attributes:
                # Find what attribute value was used for node
                attribute_value = getattr(node, attribute.name)
                # Retrieve pheromone for that value
                old_pheromone_value = attribute.dict[attribute_value]
                # Update pheromone
                attribute.dict[attribute_value] = update_rule(
                    old_value=old_pheromone_value,
                    cost=ant.cost
                )

            # Advance the current node
            current_node = neigh

    def local_update(self, old_value: float, cost):
        """Performs local pheromone update."""

        decay = cfg['aco']['pheromone']['decay']
        pheromone_0 = cfg['aco']['pheromone']['start']
        
        # print((1 - decay) * old_value + ((cost - self.best_ant.cost)))
        
        # E = (cost - self.best_ant.cost) / 10
        # pheromone_laid = 1 / E
        # new_value = (1 - decay) * old_value + pheromone_laid
        
        # print (f'decay: {decay}, old_value: {old_value}, pheromone_0: {pheromone_0}')
        return (1 - decay) * old_value + (decay * pheromone_0)

    def global_update(self, old_value: float , cost):
        """Performs global pheromone update."""

        # Calculate solution cost based on metrics
        added_pheromone = (1 / (cost * 10)) if cfg['metrics'] == 'loss' else cost
        evaporation = cfg['aco']['pheromone']['evaporation']
        return (1 - evaporation) * old_value + (evaporation * added_pheromone)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['backend']
        return d


class Ant:
    """Class responsible for representing the ant."""

    def __init__(self, path: list = []):
        self.path = path
        self.loss = math.inf
        self.accuracy = 0.0
        self.path_description = None
        self.path_hash = None

    def evaluate(self, backend, storage):
        """Evaluates how good ant's path is.

        Args:
            backend: Backend object.
            storage: Storage object.
        """
        # Extract path information
        self.path_description, path_hashes = storage.hash_path(self.path)
        self.path_hash = path_hashes[-1]

        # Check if the model already exists if yes, then just re-use it
        existing_model, existing_model_hash = storage.load_model(backend, path_hashes, self.path)
        if existing_model is None:
            # Generate model
            new_model = backend.generate_model(self.path)[0]
        else:
            # Re-use model
            new_model = existing_model

        #TODO LOW debug layers shape
        layers = ""
        for layer in new_model.layers:
            layers += f'{str(layer)}: {str(layer.input_shape)} ---> {str(layer.output_shape)} \n'
        # Log.debug(layers)

        # Train model
        new_model = backend.train_model(new_model)
        # Evaluate model
        self.loss, self.accuracy = backend.evaluate_model(new_model)

        # If the new model was created from the older model, record older model progress
        if existing_model_hash is not None:
            storage.record_model_performance(existing_model_hash, self.cost)

        # Save model
        storage.save_model(backend, new_model, path_hashes, self.cost)

    @property
    def cost(self):
        """Returns value which represents ant's cost."""

        return self.loss if cfg['metrics'] == 'loss' else self.accuracy

    def __lt__(self, other):
        return self.cost < other.cost

    def __str__(self):
        return "======= \n Ant: %s \n Loss: %f \n Accuracy: %f \n Path: %s \n Hash: %s \n=======" % (
            hex(id(self)),
            self.loss,
            self.accuracy,
            self.path_description,
            self.path_hash,
        )

class Graph:
    """Class responsible for representing the graph."""

    def __init__(self, current_depth: int = cfg['start_depth']):
        self.topology = []
        self.current_depth = current_depth
        self.input_node = self.get_node(Node.create_using_type('Input', 0))
        self.increase_depth()

    def get_node(self, node: Node):
        """Tries to retrieve a given node from the graph. If the node does not
        exist then the node is inserted into the graph before being retrieved.

        Args:
            node: Node which should be found in the graph.
        """

        # If we are trying to insert the node into a not existing layer, we pad the
        # topology by adding empty dictionaries, until the required depth is reached
        while node.depth > (len(self.topology) - 1):
            self.topology.append({})

        # If the node already exists return it, otherwise add it to the topology first
        return self.topology[node.depth].setdefault(node.name, node)

    def get_node_by_name_and_depth(self, name: str, depth: int) -> Node:
        return self.topology[depth].get(name)

    def increase_depth(self):
        """Increases the depth of the graph."""

        self.current_depth += 1

    def generate_path(self, select_rule):
        """Generates path through the graph based on given selection rule.

        Args:
            select_rule ([NeigbourNode]): function which receives a list of
            neighbours.

        Returns:
            a path which contains Node objects.
        """

        current_node = self.input_node
        path = [current_node.create_deepcopy()]
        while current_node.depth < self.current_depth: 
            # print(getsize(self.topology))
            
            # If the node doesn't have any neighbours stop expanding the path
            if not self.has_neighbours(current_node, current_node.depth):
                Log.warning('OutputNode reached before complete_path')
                break
            # Select node using given rule
            current_node = select_rule(current_node)
            # Log.debug(f'CURRENT_NODE: {current_node} with size {getsize(current_node)}') # TODO LOW debug
            # Add only the copy of the node, so that original stays unmodified
            path.append(current_node.create_deepcopy())
        completed_path = self.complete_path(path)
        return completed_path

    def has_neighbours(self, current_node: Node, depth: int):
        """Checks if the node has any neighbours.

        Args:
            node: Node that needs to be checked.
            depth: depth at which the node is stored in the graph.

        Returns:
            a boolean value which indicates if the node has any neighbours.
        """

        # neighbours_str = "" # TODO LOW debug
        # for n in current_node.neighbours:
        #     neighbours_str += f'BEFORE {current_node.name} HasNeigh : {str(n.node)}, {n.find_parent(current_node)}\n'
        #     parents_str = ""
        #     for p in n.parents:
        #         parents_str += f'Neighbour: {n.node.name} {n.node.depth} - {p}\n'
        #     Log.debug(parents_str)
        # Log.debug(neighbours_str)

        # Expand only if :
        # it hasn't been expanded
        # if the node hasn't been residualy expanded during the same depth (otherwise create duplicates)
        # if the node is eligible for new residual connections
        if  (current_node.is_expanded is False or (self.current_depth - current_node.depth <= cfg['residual_depth'] + 1)):

            #list of nodes to parse
            #only current_node for plain connections
            #current_node and its recursive neighbours for residual connections
            nodes = []
            nodes.append(current_node)

            max_residual_depth = depth + 2 + cfg['residual_depth']
            max_depth = self.current_depth + 1 if  max_residual_depth > self.current_depth else max_residual_depth

            for residual_depth in range(depth + 1, max_depth):
                temp_nodes = []

                for node in nodes: # TODO LOW if skip more than 1 layer, need to be optimized,
                    if type(node) == NeighbourNode:
                        node = self.get_node_by_name_and_depth(node.name, node.depth)
                    available_transitions = node.available_transitions
                    for (transition_name, heuristic_value) in available_transitions:
                        neighbour_node = self.get_node(Node(transition_name, residual_depth))
                        neighbour = NeighbourNode(node=neighbour_node)

                        if current_node != node:
                            if not node.find_node_into_neighbours(neighbour, heuristic_value):
                                neighbour.parents.append(ParentNode(node, heuristic=heuristic_value))
                                node.neighbours.append(neighbour)
                        if not current_node.find_node_into_neighbours(neighbour, heuristic_value):
                            neighbour.parents.append(ParentNode(current_node, heuristic=heuristic_value)) #TODO HIGH review heuristic_value for skip-connections
                            current_node.neighbours.append(neighbour)
                    temp_nodes.extend([n for n in node.neighbours if n.depth == node.depth + 1])
                nodes = temp_nodes
            current_node.is_expanded = True
            # neighbours_str = "" # TODO LOW debug
            # for n in current_node.neighbours:
            #     neighbours_str += f'AFTER {current_node.name} HasNeigh : {str(n)}, {n.find_parent(current_node)}\n'
            #     parents_str = ""
            #     for p in n.parents:
            #         parents_str += f'Neighbour: {n.name} {n.depth} - {p}\n'
            #     Log.debug(parents_str)
            # Log.debug(neighbours_str)
        # Return value indicating if the node has neighbours after being expanded
        return len(current_node.neighbours) > 0

    def complete_path(self, path: list):
        """Completes the path if it is not fully completed (i.e. missing OutputNode).

        Args:
            path [Node]: list of nodes defining the path.

        Returns:
            completed path which contains list of nodes.
        """

        # If the path is not completed, then complete it and return completed path
        # We intentionally don't add these ending nodes as neighbours to the last node
        # in the path, because during the first few iterations these nodes will always be part
        # of the best path (as it's impossible to close path automatically when it's so short)
        # this would result in bias pheromone received by these nodes during later iterations
        
        if not any(node.type == "Flatten" for node in path):
            Log.warning("Model without FlattenNode")
            if path[-1].name == 'OutputNode':
                path[-1] = self.get_node(Node.create_using_type('Flatten', path[-1].depth))
            else: 
                path.append(self.get_node(Node.create_using_type('Flatten', path[-1].depth + 1)))
        if path[-1].name in cfg['spatial_nodes']:
            Log.warning("Last Node is Spatial, add FlattenNode")
            path.append(self.get_node(Node.create_using_type('Flatten', path[-1].depth + 1)))
        if path[-1].name in cfg['flat_nodes']:
            Log.warning("Model without OutputNode")
            path.append(self.get_node(Node.create_using_type('Output', path[-1].depth + 1)))
        return path

    def show_pheromone(self):
        """Logs the pheromone information for the graph."""

        # If the output is disabled by the user then don't log the pheromone
        if cfg['aco']['pheromone']['verbose'] is False:
            return

        Log.header("PHEROMONE START", type="RED")
        for idx, layer in enumerate(self.topology):
            info = []
            for node in layer.values():
                for neighbour in node.neighbours:
                    info.append("%s [%s] -> %f -> %s [%s]" % (node.name, hex(id(node)),
                        neighbour.pheromone, neighbour.node.name, hex(id(neighbour.node))))

                    # If neighbour node doesn't have any attributes skip attribute info
                    if not neighbour.node.attributes:
                        continue

                    info.append("\t%s [%s]:" % (neighbour.node.name, hex(id(neighbour.node))))
                    for attribute in neighbour.node.attributes:
                        info.append("\t\t%s: %s" % (attribute.name, attribute.dict))
            if info:
                Log.header("Layer %d" % (idx + 1))
                Log.info('\n'.join(info))
        Log.header("PHEROMONE END", type="RED")