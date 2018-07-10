from neat import Population
from neat.population import CompleteExtinctionException
from neat.six_util import iteritems, itervalues


class AsyncPopulation(Population):

    def __init__(self, config, initial_state=None):
        super(AsyncPopulation, self).__init__(config, initial_state)

    def get_genomes(self):
        self.reporters.start_generation(self.generation)
        return list(iteritems(self.population))


    def is_complete(self, best):
        if not self.config.no_fitness_termination:
            # End if the fitness threshold is reached.
            fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
            if fv >= self.config.fitness_threshold:
                self.reporters.found_solution(self.config, self.generation, best)
                return True

    def iterate_generation(self):
        # Gather and report statistics.
        best = None
        for g in itervalues(self.population):
            if best is None or g.fitness > best.fitness:
                best = g
        self.reporters.post_evaluate(self.config, self.population, self.species, best)

        # Track the best genome ever seen.
        if self.best_genome is None or best.fitness > self.best_genome.fitness:
            self.best_genome = best

        # Create the next generation from the current generation.
        self.population = self.reproduction.reproduce(self.config, self.species,
                                                      self.config.pop_size, self.generation)

        # Check for complete extinction.
        if not self.species.species:
            self.reporters.complete_extinction()

            # If requested by the user, create a completely new population,
            # otherwise raise an exception.
            if self.config.reset_on_extinction:
                self.population = self.reproduction.create_new(self.config.genome_type,
                                                               self.config.genome_config,
                                                               self.config.pop_size)
            else:
                raise CompleteExtinctionException()

        # Divide the new population into species.
        self.species.speciate(self.config, self.population, self.generation)

        self.reporters.end_generation(self.config, self.population, self.species)

        self.generation += 1
