
# Imports

import Base.copy
using PyPlot
using ODE: ode23, ode45

# Define types

type Population
    groups::Array{Int64, 1}
    labels::Array{ASCIIString, 1}
    totalPop::Int64

    function Population(groups::Array{Int64, 1}, labels::Array{ASCIIString, 1}, totalPop::Int64)
        if sum(groups) != totalPop
            throw(ArgumentError("total population does not match sum of population vector"))
        elseif totalPop == 0
            throw(ArgumentError("population must be nonzero"))
        elseif abs(groups) != groups
            throw(ArgumentError("groups must be positive integers"))
        elseif size(labels) != size(groups)
            throw(ArgumentError("number of labels must match number of groups"))
        else
            return new(groups, labels, totalPop)
        end
    end
end

type NormalGame
    players::Int64
    strategies
    labels
    payoffFunctions::Array{Function, 1}

    function NormalGame(players::Int64, strategies, labels, payoffFunctions::Array{Function, 1})
        # Check that the number of players matches the number of strategy sets
        if size(strategies,1) != players
            throw(ArgumentError("number of players does not match size of strategies array"))
        # Check that the number of players matches the number of payoff functions
        elseif size(payoffFunction,1) != players
            throw(ArgumentError("number of players does not match size of payoff function array"))
        else
            # Check that the size of each set matches the number of labels
            for i in 1:size(strategies, 1)
                if size(strategies[i]) != size(labels[i])
                    throw(ArgumentError("not enough labels provided for each strategy"))
                end
            end
        end

        return new(players, strategies, labels, payoffFunctions)
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
        end
        if typeof(payoffStructure) <: Array
            # If the payoff matrix has negative values, we want to remap those values to positve values
            # The method by which this is done is likely to change
            if (abs(payoffStructure) != payoffStructure) && (intensityOfSelectionMap == "lin")
                throw(ArgumentError(("Negative payoffs cannot be used with linear intensity of selection mapping")))
            end
        end

        return new(population, mutationRate, payoffStructure, intensityOfSelection, intensityOfSelectionMap)
    end
end

# Constructors

function Population(groups::Array{Int64, 1}, labels::Array{ASCIIString, 1})
    totalPop = sum(groups)
    return Population(groups, labels, totalPop)
end

function Population(groups::Array{Int64,1})
    return Population(groups, ["" for i in 1:length(groups)])
end

# Copy methods

copy(arg::Population) = Population(copy(arg.groups), copy(arg.labels))

copy(arg::MoranProcess) = MoranProcess(copy(arg.population), arg.mutationRate, arg.payoffStructure, arg.intensityOfSelection, arg.intensityOfSelectionMap)

#####################################################################
# Finite population 
#####################################################################

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

function generateTimeSeries(iterations::Int64, process::MoranProcess)

    # Set up a variable to hold the time series
    timeSeries = zeros(Int64, (iterations, length(process.population.groups)))

    for i = 1:iterations
        timeSeries[i,:] = moranProcessStep!(process).population.groups
    end

    return timeSeries
end

function generateStateHistogram(iterations::Int64, process::MoranProcess)

    # If the process's population doesn't have exactly two groups, throw an error
    if length(process.population.groups) != 2
        throw(ArgumentError("Number of groups is not two and cannot be displayed as a histogram"))
    end

    # Create an empty array
    histogram = zeros(Int64, process.population.totalPop + 1)

    # Generate the timeseries
    timeSeries = generateTimeSeries(iterations, process)

    # Loop over the lines in the timeseries and add entries in the histogram
    for i in 1:size(timeSeries, 1)
        histogram[timeSeries[i,1] + 1] += 1
    end

    return histogram

end

function generateStateHeatmap(iterations::Int64, process::MoranProcess)

    # If the process's population doesn't have three groups, raise an error
    if length(process.population.groups) != 3
        throw(ArgumentError("Number of groups exceeds three and cannot be displayed as a 2D heatmap"))
    end

    # Create an empty array to hold the values
    heatmap = zeros(Int64, (process.population.totalPop + 1, process.population.totalPop + 1))

    # Generate the timeseries
    timeSeries = generateTimeSeries(iterations, process)

    # Loop over the lines in the timeseries and add corresponding entries in the matrix
    for i in 1:size(timeSeries, 1)
        heatmap[timeSeries[i,1] + 1, timeSeries[i,2] + 1] += 1 
    end

    return heatmap
end

function logNormHeatmap(heatmap::Array{Int64,2})

    # Takes a heatmap and normalises it logarithmically
    # Create a new heatmap
    newHeatmap = zeros(Float64, size(heatmap))
    # If the value is zero, make it 1
    for i in 1:size(heatmap,1)
        for j in 1:size(heatmap,2)
            if heatmap[i,j] == 0
                newHeatmap[i,j] = 1
            else
                newHeatmap[i,j] = heatmap[i,j]
            end
        end
    end

    # Take the log of all the values
    newHeatmap = log(newHeatmap)

    return newHeatmap
end

function estimateStationaryDistribution(iterations::Int64, process::MoranProcess)

    stationaryDist = zeros(Int64, length(process.population.groups))

    # Take a copy of the process to avoid destroying the original
    copyOfProcess = copy(process)

    timeSeries = generateTimeSeries(iterations, copyOfProcess)
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

# Plotting related methods

########################################################
# Infinite population methods
#######################################################

function replicator{T<:Real}(timeRange::Any, frequency::Array{T,1}, game::Array{T,2};mutationProbs = [0 0 0; 0 0 0; 0 0 0])

    """
    Parameters
    ----------
    timeRange: vector of the time range
    frequency: vector of frequencies for each strategy
    game: 2D payoff matrix
    mutationProbs: this is not used by the function but appears for compatibility with the replicator-mutator function

    Returns:
    vector of dfrequency/dt
    """
    fitness = game * frequency
    averageFitness = dot(frequency, fitness)
    #if a rounding error has put the trajectory out of bounds, stop the trajectory.
    #note: I suspect there is a way to stop the ODE integrator when the output of this
    #function is sufficiently small but I will need to look closer at the documentation
    #if frequency[1] + frequency[2] + frequency[3] > 1
    #    return [0; 0; 0]
    #end
    return frequency .* (fitness - averageFitness)
end

function replicatorTest()
    frequency = Float64[0.3; 0.7]
    game = Float64[2 1; 3 -1]
    derivative = replicator(0,frequency, game)
    epsilon = 10^(-5.0)
    if abs(derivative[1]  - 0.231) < epsilon && abs(derivative[2]  + 0.231) < epsilon
        return "replicator works properly"
    else
        return "replicator isn't working"
    end
end

function buildMutationMatrix{T<:Real}(μ::T,nStrategies::Int64)
    #build the matrix holding the mutation probabilities
    mutationProbs = fill(μ,(nStrategies,nStrategies))

    #change the diagonal so that the matrix is a stochastic matrix
    for i = 1:nStrategies
        mutationProbs[i,i] = 1 - (nStrategies - 1) * μ
    end
    return mutationProbs
end

function testBuiltMutationMatrix()
    if buildMutationMatrix(0.1,3) == [0.8 0.1 0.1; 0.1 0.8 0.1; 0.1 0.1 0.8]
        return "builtMutationMatrix works"
    else
        return "builtMutationMatrix doesn't work"
    end
end

function replicatorMutator{T<:Real}(timeRange::Any, frequency::Array{T,1},game::Array{T,2}; mutationProbs = [0 0 0; 0 0 0; 0 0 0])
    """
    Parameters
    ----------
    timeRange: vector of the time range. Only placed as the first argument for compatibility with ODE
    frequency: vector of frequencies for each strategy
    game: 2D payoff matrix
    mutationProbs:  2D mutation probability matrix

    Returns:
    vector of dfrequency/dt
    """

    fitness = game * frequency

    averageFitness = dot(frequency, fitness)

    return (mutationProbs * (frequency .* fitness)) - (averageFitness * frequency)
end

function replicatorMutatorTest()
    frequency = Float64[0.3; 0.7]
    game = Float64[2 1; 3 -1]
    mutationProbs = [0.9 0.1; 0.1 0.9]
    epsilon = 10^(-5.0)
    derivative = replicatorMutator(false,frequency, game, mutationProbs = mutationProbs)

    if abs(derivative[1]  - 0.206) < epsilon && abs(derivative[2]  + 0.206) < epsilon
        return "replicator-mutator works properly"
    else
        return "replicator-mutator isn't working"
    end

end

function considerMutation{T<:Real}(mutationProbs::Array{T,2}, μ::Float64, nStrategies::Int64)

    #if you've passed a mutationProbs matrix then we know that the replicator-mutator must be used
    if mutationProbs != [0 0 0; 0 0 0; 0 0 0]
        return mutation, replicatorMutator
    end

    #if you've pased the mutation constant then we know that a mutationProbs matrix must be built and the replicator-mutator must be used
    if μ != 0.0
        mutationProbs = buildMutationMatrix(μ,nStrategies)
        return mutationProbs, replicatorMutator
    end

    #if you haven't passed either of those then you want the replicator
    return false, replicator
end

function getTrajectory{T<:Real}(timeRange::Any,initialFrequency::Array{T,1},game::Array{T,2};mutationProbs = [0 0 0; 0 0 0; 0 0 0], μ = 0.0,solver = ode23)
    """
    Parameters
    ----------
    timeRange: vector of the time range
    initialFrequency: vector of starting frequencies for each strategy
    game: square matrix holding the payoffs of the game
    mutationProbs: 2D mutation probability matrix
    μ: mutation constant, in case you haven't premade the mutation matrix and want it done here
    solver: one of ode23, ode45, ode78 etc from the ODE.jl package

    Returns:
    a vector of time steps and a matrix of the frequencies at each time step with rows corresponding_
    to time steps and columns corresponding to strategies
    """

    nStrategies = length(initialFrequency)

    #check whether the replicator-mutator function or replicator function will be used
    mutationProbs, evoFunction = considerMutation(mutationProbs, μ, nStrategies)

    #get table of frequencies i.e. the trajectory
    timeTable, trajectory = solver((timeRange,initialFrequency) -> evoFunction(timeRange,initialFrequency,game,mutationProbs = mutationProbs), initialFrequency, timeRange)

    #number of steps used in the ODE solver
    steps = size(trajectory,1)

    #array gymnastics to turn the outputted 1D trajectory array into a 2D array
    trajectory = transpose(reshape(vcat(trajectory...),(nStrategies,steps)))

    return (timeTable, trajectory)

end

function twoStratsPhaseDiagram{T<:Real}(game::Array{T,2}; mutationProbs = [0 0 0; 0 0 0; 0 0 0], μ = 0.0, labels = ["S1", "S2"], step = 0.001)

    nStrategies = 2

    #check whether the replicator-mutator function or replicator function will be used
    mutationProbs, evoFunction = considerMutation(mutationProbs, μ, nStrategies)

    #set up the array to hold all the values of the derivative of the first strategy with respect to time
    derivative = fill(0.0,int(ceil(1/step)+1))

    #the derivative array is plotted against this so that the x-axis scales to a 0 to 1 range
    #at some point I want to find a way to bypass this because it probably wastes resources
    xaxis = linspace(0,1,length(derivative))

    #draw a line to represent the 1-simplex
    plot([0, 1], [0, 0],color = "k")

    #kwargs for the arrow function
    opt = :(head_width=0.01, head_length=0.02, width = 0.0001,fc = "b", ec = "b",length_includes_head = true)

    #record the derivative of the first strategy at each step along the range of population states, and check critical points
    j = 1
    for i = 0.0:step:1.0
        frequency = [i,1-i]

        #evoFunction will return a vector of both strategies' frequencies and we select the first one and store it
        derivative[j] = evoFunction(false, frequency, game, mutationProbs = mutationProbs)[1]

        #check whether there is a critical point and whether it is stable or unstable
        if j > 2 && j < length(derivative)
            #if a point is critical and the slope is positive i.e. it's unstable
            if derivative[j-1] > 0 && derivative[j-2] < 0 || (derivative[j-1] == 0 && derivative[j] > 0 && derivative[j-2] < 0)
                #plot a white 'o' marker i.e. an open circle
                plot(xaxis[j],0.0,"wo")
                #show arrows pointed away from the critical point
                arrow(xaxis[j],0.0,0.05,0.0,opt)
                arrow(xaxis[j],0.0,-0.05,0.0,opt)
                #if a point is critical  and the slope is negative i.e. it's stable
            elseif derivative[j-1] < 0 && derivative[j-2] > 0 || (derivative[j-1] == 0 && derivative[j] < 0 && derivative[j-2] > 0)
                #plot a black 'o' marker i.e. a closed circle
                plot(xaxis[j],0.0,"bo")
                #show arrows facing the critical point
                arrow(xaxis[j]-0.1,0.0,0.05,0.0,opt)
                arrow(xaxis[j]+0.1,0.0,-0.05,0.0,opt)
            end
        end
        j += 1
    end

    #note: need to make the arrows scale better

    #check the stability of the left end point
    if derivative[1] > 0 || (derivative[1] == 0 && derivative[2] > 0)
        plot(xaxis[1],0.0,"wo")
        arrow(xaxis[1],0.0,0.05,0.0,opt)
    elseif derivative[1] < 0 || (derivative[1] == 0 && derivative[2] < 0)
        plot(xaxis[1],0.0,"bo")
        arrow(xaxis[1]+0.1,0.0,-0.05,0.0,opt)
    end

    #check the stability of the right end point
    if derivative[length(derivative)] > 0 || (derivative[length(derivative)] == 0 && derivative[length(derivative)-1] > 0)
        plot(xaxis[length(derivative)],0.0,"bo")
        arrow(xaxis[length(derivative)]-0.1,0.0,0.05,0.0,opt)
    elseif derivative[length(derivative)] < 0 || (derivative[length(derivative)] == 0 && derivative[length(derivative)-1] < 0)
        plot(xaxis[length(derivative)],0.0,"wo")
        arrow(xaxis[length(derivative)],0.0,-0.05,0.0,opt)
    end

    plot(xaxis,derivative)

    #show the labels of each strategy
    text(-0.05,0,labels[2], ha = "right", va = "center")
    text(1.05,0,labels[1], ha = "left", va = "center")

    #enforce the x-range
    PyPlot.xlim(-0.2,1.2)
end

function threeStratsPhaseDiagram{T<:Real}(timeRange::Any, initialFrequency::Array{T,1}, game::Array{T,2}; mutationProbs =  [0 0 0; 0 0 0; 0 0 0], μ = 0.0, solver = ode23, labels = ["S1", "S2","S3"], internal = false)

    #get the trajectory
    timeTable, trajectory = getTrajectory(timeRange, initialFrequency, game, mutationProbs = mutationProbs, μ = μ, solver = solver)

    #convert the trajectory from barycentrix coordinates to cartesian coordinates
    for i = 1:size(trajectory,1)
        trajectory[i,2] = 0.8660254*trajectory[i,1]
        trajectory[i,1] = 0.5*trajectory[i,1] + trajectory[i,3]
    end

    #plot the trajectory as a phase diagram on a 2-simplex
    plot(trajectory[:,1],trajectory[:,2],0,1)

    #draw an arrow at the beginning of the trajectory indicating the direction of the trajectory
    arrow(trajectory[1,1],trajectory[1,2],trajectory[2,1]-trajectory[1,1],trajectory[2,2]-trajectory[1,2],head_width=0.02, head_length=0.02,alpha = 0.3, fc = "k", ec = "k")

    if internal == false
        #show the labels of each strategy
        if labels != ["","",""]
            text(-0.1,-0.1,labels[2], ha = "center")
            text(1.1,-0.1,labels[3], ha = "center")
            text(0.5,1.0,labels[1], ha = "center")
        end

        #enforce the x and y ranges and remove the axes
        ylim(-0.2,1.2)
        xlim(-0.2,1.2)
        axis("off")

        #plot the simplex edges
        plot([0, 0.5], [0, 0.8660254],color = "k")
        plot([1, 0.5], [0, 0.8660254],color = "k")
        plot([0, 1], [0, 0],color = "k")
    end
end

function plotThreeStratsMultiTrajectories{T<:Real}(timeRange::Array{T,1}, game::Array{T,2}; mutationProbs = [0 0 0; 0 0 0; 0 0 0], μ = 0.0, step = 0.2,solver = ode23,labels = ["S1","S2","S3"])
    #plot trajectories starting at different points, determined by the value of step
    for i = 0.0:step:1.0
        for j = 0.0:step:(1-i)
            initialFrequency = [i; j; 1-i-j]
            threeStratsPhaseDiagram(timeRange,initialFrequency,game,mutationProbs = mutationProbs, μ = μ, labels = ["","",""], internal = true)
        end
    end

    #show the labels of each strategy
    text(-0.1,-0.1,labels[2],ha = "center")
    text(1.1,-0.1,labels[3],ha = "center")
    text(0.5,1.0,labels[1],ha = "center")

    #enforce the x and y ranges and remove the axes
    ylim(-0.2,1.2)
    xlim(-0.2,1.2)
    axis("off")

    #plot the simplex edges
    plot([0, 0.5], [0, 0.8660254],color = "k")
    plot([1, 0.5], [0, 0.8660254],color = "k")
    plot([0, 1], [0, 0],color = "k")

end

function plotThreeStratsVectorField{T<:Real}(game::Array{T,2}; mutationProbs = [0 0 0; 0 0 0; 0 0 0], μ = 0.0,step = 0.1)

    nStrategies = size(game,1)

    #determine whether the replicator-mutator or replicator will be used and whether a mutationProbs matrix needs to be constructed
    mutationProbs, evoFunction = considerMutation(mutationProbs, μ, nStrategies)
    maxStrength = 0
    k = 1
    vectors= fill(0.0,(int(floor(1.0/step+1)*(floor(1.0/step+2))/2),2))
    positions =  fill(0.0,(int(floor(1.0/step+1)*(floor(1.0/step+2))/2),2))

    #draw a field of vectors indicating the direction of trajectories at different points, determined by the value of step
    for i = 0.0:step:1.0
        for j = 0.0:step:(1-i)

            frequency = [i; j; 1-i-j]
            #get X dot
            frequencyDot = evoFunction([0,0], frequency,game,mutationProbs = mutationProbs)

            vectors[k,1] = frequencyDot[3]
            vectors[k,2] = frequencyDot[1]
            positions[k,1] = 1-i-j
            positions[k,2] = i

            k += 1
        end
    end

    #convert from barycentric coordinates to cartesian
    vectorsCart = fill(0.0,(k,2))
    positionsCart = fill(0.0,(k,2))
    for i = 1:k
        vectorsCart[i,1] = 0.5*vectors[i,2] + vectors[i,1]
        vectorsCart[i,2] = 0.8660254*vectors[i,2]
        positionsCart[i,1] = 0.5*positions[i,2] + positions[i,1]
        positionsCart[i,2] = 0.8660254*positions[i,2]
    end

    #draw the vectors
    quiver(positionsCart[:,1],positionsCart[:,2],vectorsCart[:,1],vectorsCart[:,2])

    #show the labels of each strategy
    text(-0.1,-0.1,labels[2],ha = "center")
    text(1.1,-0.1,labels[3],ha = "center")
    text(0.5,1.0,labels[1],ha = "center")

    #enforce the x and y ranges
    ylim(-0.2,1.2)
    xlim(-0.2,1.2)
    axis("off")

    #plot the simplex edges
    plot([0, 0.5], [0, 0.8660254],color = "k")
    plot([1, 0.5], [0, 0.8660254],color = "k")
    plot([0, 1], [0, 0],color = "k")

end

function plotAgainstTime{T<:Real}(timeRange::Any, initialFrequency::Array{T,1}, game::Array{T,2};labels = ["S1"], mutationProbs = [0 0 0; 0 0 0; 0 0 0], μ = 0.0, solver = ode23)

    nStrategies = length(initialFrequency)

    #get the frequencies of each strategy and the table of time steps taken
    timeTable, trajectory = getTrajectory(timeRange, initialFrequency, game, mutationProbs = mutationProbs, μ = μ, solver = solver)

    #make more labels if you don't have enough
    for i = length(labels):nStrategies
        labels = vcat(labels,["S$(i+1)"])
    end

    #plot each frequency against time
    for i = 1:nStrategies
        plot(timeTable,trajectory[:,i], label = labels[i])
    end

    #show the legend
    legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    #enforce x and y ranges
    PyPlot.ylim(-0.1,1.1)
    PyPlot.xlim(0,timeTable[length(timeTable)])

end

function clearSimplex()
    clf()
end
