import embod_client as client
import os
from uuid import UUID
import neat
from collections import defaultdict, Counter
from state_stack import StateStack
from AsyncPopulation import AsyncPopulation

class EvolvingController:

    def __init__(self, api_key, agent_ids, host, max_steps):

        self._api_key = api_key
        self._agent_ids = [UUID(agent_id) for agent_id in agent_ids]
        self._host = host

        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config')
        self._config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

        self._pop = AsyncPopulation(self._config)
        stats = neat.StatisticsReporter()
        self._pop.add_reporter(stats)
        self._pop.add_reporter(neat.StdOutReporter(True))
        # Checkpoint every 25 generations or 900 seconds.
        self._pop.add_reporter(neat.Checkpointer(25, None))

        self._nets = {}
        self._agent_genomes = {}
        self._states = defaultdict(lambda: StateStack(1))
        self._steps = Counter()
        self._max_steps = max_steps
        self._num_running = 0
        self._generation_complete = False

        self._genome_offset = 0
        self._is_running = defaultdict(lambda: False)

        self.client = client.AsyncClient(api_key, self._on_connect, self._state_callback, host)

    async def _state_callback(self, agent_id, state, reward, error):

        if not self._is_running[agent_id]:
            return

        self._steps[agent_id] += 1
        if self._steps[agent_id] >= self._max_steps:
            self._is_running[agent_id] = False
            self._steps[agent_id] = 0
            await self._on_complete(agent_id)
            return

        self._states[agent_id].add_state(state)

        if error is not None:
            print(error[0].decode('UTF-8'))

        if reward is not None:
            if reward > 0:
                print("Consumed :)")
                self._agent_genomes[agent_id].fitness += 1

            if reward < 0:
                print("Got Consumed :(")
                self._agent_genomes[agent_id].fitness -= 1

        nth_state = self._states[agent_id].get_state()
        if nth_state is not None:
            action = self._nets[agent_id].activate(nth_state)

            await self.client.send_agent_action(agent_id, action)


    def get_agents_to_evaluate(self):

        num_agents = len(self._agent_ids)

        configured_ids = []

        for i in range(0, num_agents):

            if not self._genomes:
                self._generation_complete = True
                break

            genome_id, genome = self._genomes.pop()

            agent_id = self._agent_ids[i % num_agents]
            self._nets[agent_id] = neat.nn.FeedForwardNetwork.create(genome, self._config)
            genome.fitness = 0.0
            self._agent_genomes[agent_id] = genome

            self._is_running[agent_id] = True
            self._num_running += 1

            configured_ids.append(agent_id)

        return configured_ids



    async def _on_connect(self):
        """
        Firsty add agents to the environment and then run evaluation cycles
        :return:
        """

        self._genomes = self._pop.get_genomes().copy()

        await self.continue_evaluation()


    async def continue_evaluation(self):
        configured_ids = self.get_agents_to_evaluate()

        for agent_id in configured_ids:
            await self.client._add_agent(agent_id)


    async def _on_complete(self, agent_id):
        await self.client._remove_agent(agent_id)
        print("removing agent %s" % agent_id)

        self._is_running[agent_id] = False

        self._num_running -= 1
        if self._num_running == 0:

            if self._generation_complete:
                self._genome_offset = 0
                self._generation_complete = False
                self._pop.iterate_generation()
                self._genomes = self._pop.get_genomes().copy()

            await self.continue_evaluation()


    def run(self):
        self.client.start()

