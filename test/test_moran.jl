module TestMoran
using Optim
using Base.Test

# Include our custom test handler
include("customTestHandler.jl")

include("../src/Jedy.jl")

Test.with_handler(customHandler) do

f(X) = (10 - X[1, 1])^2 + (0 - X[1, 2])^2 + (0 - X[2, 1])^2 + (5 - X[2, 2])^2

function g!(X, S)
    S[1, 1] = -20 + 2 * X[1, 1]
    S[1, 2] = 2 * X[1, 2]
    S[2, 1] = 2 * X[2, 1]
    S[2, 2] = -10 + 2 * X[2, 2]
    return
end

res = optimize(f, g!, eye(2), method = :gradient_descent)

@test norm(vec(res.minimum - [10.0 0.0; 0.0 5.0])) < 10e-8

# TODO: Get finite differencing to work for generic arrays as well
# optimize(f, eye(2), method = :gradient_descent)

# Test the constructor of the Population type
# Test that the correct total population is generated
testArr = [10 for _ = 1:5]
testPop = Population(testArr)
@test testPop.totalPop == 50

# Test that an error is thrown if the sum is not correct
@test_throws ArgumentError Population(testArr, 40)

# Test that an error is thrown if the total population is zero
@test_throws ArgumentError Population([0, 0, 0])

# Test that an error is thrown if the population has negative values
@test_throws ArgumentError Population([-1, 2, 0])

# Test fitness function for matrix games
# Test that the fitness is correct for a simple game and a simple population
payoffMatrix = [0 1; 1 0]
pop = Population([10,10])
@test fitness(pop, payoffMatrix) == [10/19,10/19]

end

end
