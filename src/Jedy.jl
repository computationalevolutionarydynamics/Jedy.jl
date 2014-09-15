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

# Helper methods

function sampleFromPopulations(probabilities::Array{Float64})
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
