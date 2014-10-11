# Imports

import Base.copy

# Define types

type Population
    groups::Array{Int64, 1}
    totalPop::Int64

    function Population(groups::Array{Int64, 1}, totalPop::Int64)
        if sum(groups) != totalPop
            throw(ArgumentError("total population does not match sum of population vector"))
        elseif totalPop == 0
            throw(ArgumentError("population must be nonzero"))
        elseif abs(groups) != groups
            throw(ArgumentError("groups must be positive integers"))
        else
            return new(groups, totalPop)
        end
    end
end

type MoranProcess
    population::Population
    mutationRate::Float64
    payoffStructure
    intensityOfSelection::Real
    intensityOfSelectionMap::ASCIIString

    function MoranProcess(population::Population, mutationRate::Float64, payoffStructure, intensityOfSelection::Real, intensityOfSelectionMap::ASCIIString)
        if (intensityOfSelectionMap != "lin") && (intensityOfSelectionMap != "exp")
            throw(ArgumentError("Invalid intensity of selection mapping type"))
        else
            return new(population, mutationRate, payoffStructure, intensityOfSelection, intensityOfSelectionMap)
        end
    end
end

# Constructors

function Population(groups::Array{Int64, 1})
    totalPop = sum(groups)
    return Population(groups, totalPop)
end

# Copy methods

copy(arg::Population) = Population(copy(arg.groups))

copy(arg::MoranProcess) = MoranProcess(copy(arg.population), arg.mutationRate, arg.payoffStructure, arg.intensityOfSelection, arg.intensityOfSelectionMap)

# Finite population functions

function fitness{T<:Real}(pop::Population, payoffMatrix::Array{T,2}, intensityOfSelection::T, intensityOfSelectionMap::ASCIIString)
    if (intensityOfSelectionMap != "lin") && (intensityOfSelectionMap != "exp")
        throw(ArgumentError("Invalid intensity of selection mapping type"))
    elseif intensityOfSelectionMap == "lin"
        mappedPayoff = linear_fitness_map(payoffMatrix, intensityOfSelection)
    elseif intensityOfSelectionMap == "exp"
        mappedPayoff = exponential_fitness_map(payoffMatrix, intensityOfSelection)
    end

    fitnessVector = mappedPayoff * pop.groups
    fitnessVector -= diag(mappedPayoff)
    fitnessVector /= pop.totalPop - 1
end

function reproductionProbability{T<:Real}(pop::Population, payoffMatrix::Array{T,2}, intensityOfSelection::T, intensityOfSelectionMap::ASCIIString)
    fitnessVector = fitness(pop, payoffMatrix, intensityOfSelection, intensityOfSelectionMap)
    probVector = fitnessVector .* pop.groups
    probVector /= fitnessVector ⋅ pop.groups
end

function moranProcessStep!(process::MoranProcess)
    # Get the reproduction probability distribution
    reproductionProbs = reproductionProbability(process.population, process.payoffStructure, process.intensityOfSelection, process.intensityOfSelectionMap)

    # Select the group that will reproduce
    reproductionGroup = sampleFromPDF(reproductionProbs)

    # Decide whether the offspring will mutate
    # Generate a vector where each element is the mutation rate
    mutationVector = [process.mutationRate for _ = 1:length(process.population.groups)]

    # Set the proability that the population doesn't mutate so that the probabilities sum to 1
    mutationVector[reproductionGroup] = 1 - sum(mutationVector) + process.mutationRate

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

function estimateTimeSeries(iterations::Int64, process::MoranProcess)

    # Set up a variable to hold the time series
    timeSeries = Array(Int64, (iterations, length(process.population.groups)))

    for i = 1:iterations
        timeSeries[i,:] = moranProcessStep!(process).population.groups
    end

    return timeSeries
end

function estimateStationaryDistribution(iterations::Int64, process::MoranProcess)

    stationaryDist = Array(Int64, length(process.population.groups))

    # Take a copy of the process to avoid destroying the original
    copyOfProcess = copy(process)

    timeSeries = estimateTimeSeries(iterations, copyOfProcess)
    for i in 1:size(timeSeries, 1)
        for j in 1:size(timeSeries, 2)
            if timeSeries[i, j] == copyOfProcess.population.totalPop
                stationaryDist[j] += 1
            end
        end
    end

    # Divide by the number of entries
    stationaryDist /= sum(stationaryDist)
end

function computeFixationProbability{T<:Real}(payoffMatrix::Array{T,2}, dominantPop::Int64, mutantPop::Int64, mutantSize::Int64, totalPopSize::Int64,
                                            intensityOfSelection::T, intensityOfSelectionMap::ASCIIString)
    
    numGroups = size(payoffMatrix,1)
    gamma = zeros(Float64, totalPopSize - 1)
            
    # Loop over all the pop sizes
    for k = 1:totalPopSize - 1
            
        # Generate the population
        popArray = zeros(Int64, numGroups)
        popArray[dominantPop] = totalPopSize - k
        popArray[mutantPop] = k
        pop = Population(popArray)

        # Find the reproduction probabilities
        reproductionProbs = reproductionProbability(pop, payoffMatrix, intensityOfSelection, intensityOfSelectionMap)

        # Figure out the probability of mutant decreasing and prob of mutant increasing
        probDecrease = reproductionProbs[dominantPop] * k / totalPopSize
        probIncrease = reproductionProbs[mutantPop] * (totalPopSize - k) / totalPopSize
                
        # Calculate gamma
        gamma[k] = probDecrease/probIncrease
    end
            
    # Now calculate the fixation probability
    fixationProbability =  (1 + sum(map((x)->prod(gamma[1:x]),[1:mutantSize - 1]))) / (1 + sum(map((x)->prod(gamma[1:x]),[1:totalPopSize-1])))
end

function computeTransitionMatrix(process::MoranProcess)
    
    # Get the number of groups
    numGroups = size(process.payoffStructure, 1)
    
    transitionMatrix = zeros(Float64, (numGroups,numGroups))
    
    # Loop over the groups
    for i = 1:numGroups
        
        # Loop over the groups excluding the combination with itself
        for j = [1:i-1, i+1:numGroups]
            
            transitionMatrix[i,j] = computeFixationProbability(process.payoffStructure, i, j, 1, process.population.totalPop,
                                                                process.intensityOfSelection, process.intensityOfSelectionMap)

        end
        
        # Calculate the probability on the diagonal by ensuring that the matrix is stochastic
        transitionMatrix[i, i] = 1 - sum(transitionMatrix[i, :])
    end
    
    return transitionMatrix
end

function computeStationaryDistribution(process::MoranProcess)
    
    transitionMatrix = computeTransitionMatrix(process)
    stationaryVector = abs(eig(transitionMatrix)[2][2,:])
    stationaryVector /= sum(stationaryVector)
    
end


# Fitness mapping

# Array value
exponential_fitness_map{T<:Real}(payoff::Array{T}, intensityOfSelection::T) = exp(intensityOfSelection*payoff)

# Single value
exponential_fitness_map{T<:Real}(payoff::T, intensityOfSelection::T) = exp(intensityOfSelection*payoff)

# Array value
linear_fitness_map{T<:Real}(payoff::Array{T}, intensityOfSelection::T) = 1 - intensityOfSelection + intensityOfSelection*payoff

# Single value
linear_fitness_map{T<:Real}(payoff::T, intensityOfSelection::T) = 1 - intensityOfSelection + intensityOfSelection*payoff


# Helper methods

function sampleFromPDF(probabilities::Array{Float64})
    if abs(sum(probabilities) - 1.0) > 1e-2
      throw(ArgumentError("probabilities must add to 1"))
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


