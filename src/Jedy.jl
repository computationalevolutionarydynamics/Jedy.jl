# Imports

import Base.copy

# Define types

type Population
    groups::Array{Int64, 1}
    totalPop::Int64

    function Population(groups::Array{Int64, 1}, totalPop::Int64)
        if sum(groups) != totalPop
            error("total population does not match sum of population vector")
        else
            return new(groups, totalPop)
        end
    end
end

type MoranProcess
    population::Population
    mutationRate::Float64
    payoffStructure
    intensityOfSelection::Float64
end

# Constructors

function Population(groups::Array{Int64, 1})
    totalPop = sum(groups)
    return Population(groups, totalPop)
end

# Copy methods

copy(arg::Population) = Population(copy(arg.groups))

# Finite population functions

function fitness(pop::Population, payoffMatrix::Array{Float64,2})
    fitnessVector = payoffMatrix * pop.groups
    fitnessVector -= diag(payoffMatrix)
    fitnessVector /= pop.totalPop - 1
end

function reproductionProbability(pop::Population, payoffMatrix::Array{Float64,2})
    fitnessVector = fitness(pop, payoffMatrix)
    probVector = fitnessVector .* pop.groups
    probVector /= fitnessVector ⋅ pop.groups
end

function moranProcessStep!(process::MoranProcess)
    # Get the reproduction probability distribution
    reproductionProbs = reproductionProbability(process.population, process.payoffStructure)

    # Select the group that will reproduce
    reproductionGroup = sampleFromPDF(reproductionProbs)

    # Decide whether the offspring will mutate
    # Generate a vector where each element is the mutation rate
    mutationVector = [process.mutationRate for _ = 1:length(process.population)]

    # Set the proability that the population doesn't mutate so that the probabilities sum to 1
    mutationVector[reproductionGroup] = 1 - process.mutationRate - sum(mutationVector)

    # Sample from the mutation vector
    offspringGroup = sampleFromPDF(mutationVector)

    # Figure out the death probabilities
    deathProbs = process.population.groups / process.population.totalPop

    # Figure out the death group
    deathGroup = sampleFromPDF(deathProbs)

    # Update the population
    # If the offspringGroup = deathGroup, nothing happens
    if offspringGroup != deathGroup
        process.population.groups[offspringGroup] += 1
        process.population.groups[deathGroup] -= 1

    end

    return process
end

# Helper methods

function sampleFromPDF(probabilities::Array{Float64})
    if sum(probabilities) != 1
      error("probabilities must add to 1")
    else
        # Generate a random number
        randNum = rand()
        # Loop over elements of the input array
        for i = 1:length(probabilities)
            # Select the first value of i for which the random number is less
            # than the discrete CDF
            if randNum < sum(probabilities[1:i])
                return i
            end
        end
    end
end
