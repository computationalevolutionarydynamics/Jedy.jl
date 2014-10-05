# Define a custom error handler to give formatted success or failures
customHandler(r::Test.Success) = print_with_color(:green, "Success on $(r.expr)\n")
customHandler(r::Test.Failure) = error("$(r.expr)")
customHandler(r::Test.Error) = rethrow(r)