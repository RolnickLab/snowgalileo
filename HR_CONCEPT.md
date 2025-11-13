

# Create a random subset of test masks with pytorch subset (should be fixed, though).
# Predict the subset twice: Once with HR checkpoint, and once without.
# Make sure the test inputs have HR included and not, respectively.
# Store the prediction R2 and RMSE values, respectively, and somehow keep track of where in the dataset they occured.
# Compare the metric arrays, and extract the indeces where non-HR is worse than HR, as well as the ones where HR is much higher than non-HR.
# Visualize the results.
# For the indeces: get the respective input image and see where in the timeseries HR data occured.