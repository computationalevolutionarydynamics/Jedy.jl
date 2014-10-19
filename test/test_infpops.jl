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

function testBuiltMutationMatrix()
    if buildMutationMatrix(0.1,3) == [0.8 0.1 0.1; 0.1 0.8 0.1; 0.1 0.1 0.8]
        return "builtMutationMatrix works"
    else
        return "builtMutationMatrix doesn't work"
    end
end
