type Population
    groups::Array{Int64, 1}
    totalPop::Int64
    
    function Population(groups::Array{Int64, 1})
        totalPop = sum(groups)
        return new(groups, totalPop)
    end
end

import Base.copy

copy(arg::Population) = Population(copy(arg.groups))