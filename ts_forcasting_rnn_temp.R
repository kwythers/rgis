#################################################################
# weather timeseries dataset recorded at the Weather Station 
# at the Max Planck Institute for Biogeochemistry in 
# Jena, Germany
#################################################################

##### temperature-forecasting #####

# load libraries
library(tidyverse)
library(keras)


# Download and uncompress the data
# dir.create("~/Downloads/jena_climate", recursive = TRUE)
# download.file(
#   "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip",
#   "~/Downloads/jena_climate/jena_climate_2009_2016.csv.zip"
# )
# unzip(
#   "~/Downloads/jena_climate/jena_climate_2009_2016.csv.zip",
#   exdir = "~/Downloads/jena_climate"
# )

# load and look at the data
data_dir <- "~/R_data/jena_climate"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read_csv(fname)

glimpse(data)

# plot of temperature (in degrees Celsius) over time - on this plot, you can clearly 
ggplot(data, aes(x = 1:nrow(data), y = `T (degC)`)) + geom_line()

# a more narrow plot of the first 10 days of temperature data - because the data is recorded every 10 minutes, 
# you get 144 data points per day
ggplot(data[1:1440,], aes(x = 1:1440, y = `T (degC)`)) + geom_line()

# If you were trying to predict average temperature for the next month given a few months of past data, 
# the problem would be easy, due to the reliable year-scale periodicity of the data. But looking at the 
# data over a scale of days, the temperature looks a lot more chaotic. Is this time series predictable 
# at a daily scale? Let’s find out.

##### Preparing the data


# lookback = 1440 — Observations will go back 10 days.
# steps = 6 — Observations will be sampled at one data point per hour.
# delay = 144 — Targets will be 24 hours in the future.
# 
# To get started, you need to do two things:
#   
# Preprocess the data to a format a neural network can ingest. This is easy: the data is already numerical, so you 
# don’t need to do any vectorization. But each time series in the data is on a different scale (for example, 
# temperature is typically between -20 and +30, but atmospheric pressure, measured in mbar, is around 1,000). You’ll 
# normalize each time series independently so that they all take small values on a similar scale.
# 
# Write a generator function that takes the current array of float data and yields batches of data from the recent 
# past, along with a target temperature in the future. Because the samples in the dataset are highly redundant 
# (sample N and sample N + 1 will have most of their timesteps in common), it would be wasteful to explicitly allocate 
# every sample. Instead, you’ll generate the samples on the fly using the original data

sequence_generator <- function(start) {
  value <- start - 1
  function() {
    value <<- value + 1
    value
  }
}

gen <- sequence_generator(10)
gen()

# The current state of the generator is the value variable that is defined outside of the function. Note that 
# superassignment (<<-) is used to update this state from within the function.
# 
# Generator functions can signal completion by returning the value NULL. However, generator functions passed to 
# Keras training methods (e.g. fit_generator()) should always return values infinitely (the number of calls to the 
# generator function is controlled by the epochs and steps_per_epoch parameters)

# convert the R data frame which we read earlier into a matrix of floating point values 
# (discard the first column which included a text timestamp)

data <- data.matrix(data[,-1])

# preprocess the data by subtracting the mean of each time series and dividing by the standard deviation - use the 
# first 200,000 timesteps as training data, so compute the mean and standard deviation for normalization only on this 
# fraction of the data
train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)

# the code for the data generator you’ll use is below - yields a list (samples, targets), where samples is one batch 
# of input data and targets is the corresponding array of target temperatures. It takes the following arguments:
#   
# data — The original array of floating-point data, which you normalized
# lookback — How many timesteps back the input data should go
# delay — How many timesteps in the future the target should be
# min_index and max_index — Indices in the data array that delimit which timesteps to draw from - 
#   this is useful for keeping a segment of the data for validation and another for testing.
# shuffle — Whether to shuffle the samples or draw them in chronological order
# batch_size — The number of samples per batch
# step — period, in timesteps, at which you sample data - set it 6 in order to draw one data point every hour

generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size-1, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows),
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]]-1,
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }           
    list(samples, targets)
  }
}

# the i variable contains the state that tracks next window of data to return, so it is updated using 
# superassignment (e.g. i <<- i + length(rows))
# 
# use the abstract generator function to instantiate three generators: one for training, one for 
# validation, and one for testing. Each will look at different temporal segments of the original data: the 
# training generator looks at the first 200,000 timesteps, the validation generator looks at the following 
# 100,000, and the test generator looks at the remainder

lookback <- 1440
step <- 6
delay <- 144
batch_size <- 128

train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)

val_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step,
  batch_size = batch_size
)

test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)

# how many steps to draw from val_gen in order to see the entire validation set
val_steps <- (300000 - 200001 - lookback) / batch_size

# how many steps to draw from test_gen in order to see the entire test set
test_steps <- (nrow(data) - 300001 - lookback) / batch_size

##### A common-sense, non-machine-learning baseline 
# before using black-box deep-learning models to solve the temperature-prediction problem, let’s try a simple, common-sense 
# approach - it will serve as a sanity check, and it will establish a baseline that you’ll have to beat in order to demonstrate 
# the usefulness of more-advanced machine-learning models - such common-sense baselines can be useful when you’re approaching 
# a new problem for which there is no known solution - classic example is that of unbalanced classification tasks, where 
# some classes are much more common than others - if your dataset contains 90% instances of class A and 10% instances of class 
# B, then a common-sense approach to the classification task is to always predict “A” when presented with a new sample - such 
# a classifier is 90% accurate overall, and any learning-based approach should therefore beat this 90% score in order to 
# demonstrate usefulness - such elementary baselines can prove surprisingly hard to beat

# in this case, the temperature time series can safely be assumed to be continuous (the temperatures tomorrow are likely to 
# be close to the temperatures today) as well as periodical with a daily period. Thus a common-sense approach is to always 
# predict that the temperature 24 hours from now will be equal to the temperature right now. Let’s evaluate this approach, 
# using the mean absolute error (MAE) metric: mean(abs(preds - targets))

evaluate_naive_method <- function() {
  batch_maes <- c()
  for (step in 1:val_steps) {
    c(samples, targets) %<-% val_gen()
    preds <- samples[,dim(samples)[[2]],2]
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}

evaluate_naive_method()
# yields an MAE of 0.29. Because the temperature data has been normalized to be centered on 0 and have a standard 
# deviation of 1, this number isn’t immediately interpretable. It translates to an average absolute error of 
# 0.29 x temperature_std degrees Celsius: 2.57˚C
celsius_mae <- 0.29 * std[[2]]
# that’s a fairly large average absolute error. Now the game is to use your knowledge of deep learning to do better

##### basic machine-learning approach
# in the same way that it’s useful to establish a common-sense baseline before trying machine-learning approaches, it’s 
# useful to try simple, cheap machine-learning models (such as small, densely connected networks) before looking into 
# complicated and computationally expensive models such as RNNs - this is the best way to make sure any further complexity 
# you throw at the problem is legitimate and delivers real benefits
# 
# the following listing shows a fully connected model that starts by flattening the data and then runs it through two 
# dense layers. Note the lack of activation function on the last dense layer, which is typical for a regression 
# problem - you use MAE as the loss - because you evaluate on the exact same data and with the exact same metric 
# you did with the common-sense approach, the results will be directly comparable
model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)

# display the loss curves for validation and training
plot(history)
# some of the validation losses are close to the no-learning baseline, but not reliably. This goes to show the merit of 
# having this baseline in the first place: it turns out to be not easy to outperform. Your common sense contains a lot 
# of valuable information that a machine-learning model doesn’t have access to

##### a first recurrent baseline
# the first fully connected approach didn’t do well, but that doesn’t mean machine learning isn’t applicable to this 
# problem -the previous approach first flattened the time series, which removed the notion of time from the input data - let’s 
# instead look at the data as what it is: a sequence, where causality and order matter - try a recurrent-sequence processing 
# model – it should be the perfect fit for such sequence data, precisely because it exploits the temporal ordering of 
# data points, unlike the first approach
# 
# instead of the LSTM layer introduced in the previous section, you’ll use the GRU layer, developed by Chung et al. in 2014 - 
# gated recurrent unit (GRU) layers work using the same principle as LSTM, but they’re somewhat streamlined and thus cheaper 
# to run (although they may not have as much representational power as LSTM). This trade-off between computational expensiveness 
# and representational power is seen everywhere in machine learning

model <- keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)
plot(history)

##### using recurrent dropout to fight overfitting
# it’s evident from the training and validation curves that the model is overfitting: the training and validation losses 
# start to diverge considerably after a few epochs. You’re already familiar with a classic technique for fighting this 
# phenomenon: dropout, which randomly zeros out input units of a layer in order to break happenstance correlations in the 
# training data that the layer is exposed to. But how to correctly apply dropout in recurrent networks isn’t a trivial 
# question - it has long been known that applying dropout before a recurrent layer hinders learning rather than helping 
# with regularization. In 2015, Yarin Gal, as part of his PhD thesis on Bayesian deep learning, determined the proper 
# way to use dropout with a recurrent network: the same dropout mask (the same pattern of dropped units) should be applied 
# at every timestep, instead of a dropout mask that varies randomly from timestep to timestep. What’s more, in order to 
# regularize the representations formed by the recurrent gates of layers such as layer_gru and layer_lstm, a temporally 
# constant dropout mask should be applied to the inner recurrent activations of the layer (a recurrent dropout mask) - using 
# the same dropout mask at every timestep allows the network to properly propagate its learning error through time; a 
# temporally random dropout mask would disrupt this error signal and be harmful to the learning process
# 
# every recurrent layer in Keras has two dropout-related arguments: dropout, a float specifying the dropout rate for input 
# units of the layer, and recurrent_dropout, specifying the dropout rate of the recurrent units - let’s add dropout and 
# recurrent dropout to the layer_gru and see how doing so impacts overfitting. Because networks being regularized with 
# dropout always take longer to fully converge, you’ll train the network for twice as many epochs
model <- keras_model_sequential() %>% 
  layer_gru(units = 32, dropout = 0.2, recurrent_dropout = 0.2,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 40,
  validation_data = val_gen,
  validation_steps = val_steps
)
plot(history)


